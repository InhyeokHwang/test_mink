from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

_HERE = Path(__file__).parent
_XML = Path(__file__).parent.parent / "description" / "dual_arm" / "scene.xml"

SOLVER = "daqp"


POSTURE_COST = 1e-4

# IK 세팅
MAX_ITERS_PER_CYCLE = 20
DAMPING = 5e-4

# 목표 수렴 판정
POS_THRESHOLD = 1e-3
ORI_THRESHOLD = 1e-2

# viewer loop rate
RATE_HZ = 60.0

# mocap target 시각화
TARGET_RADIUS = 0.03
TARGET_RGBA_LEFT = [0.1, 0.9, 0.1, 0.9]
TARGET_RGBA_RIGHT = [0.1, 0.1, 0.9, 0.9]


# -------------------------
# Utilities
# -------------------------
def list_site_names(model: mujoco.MjModel) -> List[str]:
    out = []
    for i in range(model.nsite):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if nm:
            out.append(nm)
    return out


def pick_two_ee_sites(model: mujoco.MjModel) -> Tuple[str, str]:
    """
    우선순위:
      1) gripper_left / gripper_right
      2) attachment_site_left / attachment_site_right (panda 예시 스타일)
      3) 이름에 'gripper' 포함된 site 2개
      4) site 2개 이상이면 앞의 2개
    """
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_left") != -1 and \
       mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_right") != -1:
        return "gripper_left", "gripper_right"

    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_left") != -1 and \
       mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_right") != -1:
        return "attachment_site_left", "attachment_site_right"

    sites = list_site_names(model)
    grippers = [s for s in sites if "gripper" in s.lower()]
    if len(grippers) >= 2:
        return grippers[0], grippers[1]

    if len(sites) >= 2:
        return sites[0], sites[1]

    raise RuntimeError("Need 2+ sites for bimanual IK (add gripper_left/right).")


def quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> float:
    """min(||q - qt||, ||q + qt||)"""
    return min(np.linalg.norm(q_current - q_target), np.linalg.norm(q_current + q_target))


def get_site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise RuntimeError(f"site not found: {site_name}")

    pos = data.site_xpos[sid].copy()
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    return pos, quat


def check_reached(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_left: str,
    site_right: str,
    tgt_left_pos: np.ndarray,
    tgt_left_quat: np.ndarray,
    tgt_right_pos: np.ndarray,
    tgt_right_quat: np.ndarray,
    pos_th: float,
    ori_th: float,
) -> bool:
    ml_pos, ml_q = get_site_pose(model, data, site_left)
    mr_pos, mr_q = get_site_pose(model, data, site_right)

    err_pos_l = np.linalg.norm(ml_pos - tgt_left_pos)
    err_ori_l = quaternion_error(ml_q, tgt_left_quat)
    err_pos_r = np.linalg.norm(mr_pos - tgt_right_pos)
    err_ori_r = quaternion_error(mr_q, tgt_right_quat)

    return (err_pos_l <= pos_th and err_ori_l <= ori_th and err_pos_r <= pos_th and err_ori_r <= ori_th)


def _ensure_mocap_target(
    spec: mujoco.MjSpec,
    name: str,
    rgba: List[float],
) -> None:
    """worldbody에 mocap body + sphere geom 보장."""
    try:
        body = spec.body(name)
    except Exception:
        body = None

    if body is None:
        body = spec.worldbody.add_body(name=name, mocap=True)

    r = float(TARGET_RADIUS)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[r, r, r],
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )


def load_model_with_targets(xml_path: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml_path.as_posix())
    _ensure_mocap_target(spec, "target_left", TARGET_RGBA_LEFT)
    _ensure_mocap_target(spec, "target_right", TARGET_RGBA_RIGHT)
    return spec.compile()


def initialize_mocap_targets_to_sites(model: mujoco.MjModel, data: mujoco.MjData, site_left: str, site_right: str):
    """mocap target의 pos/quat을 현재 EE site와 일치시키기."""
    # ids
    sid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_left)
    sid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_right)
    if sid_l < 0 or sid_r < 0:
        raise RuntimeError("EE sites not found")

    mid_l = model.body("target_left").mocapid
    mid_r = model.body("target_right").mocapid

    # pos
    data.mocap_pos[mid_l] = data.site_xpos[sid_l].copy()
    data.mocap_pos[mid_r] = data.site_xpos[sid_r].copy()

    # quat
    ql = np.empty(4, dtype=np.float64)
    qr = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(ql, data.site_xmat[sid_l])
    mujoco.mju_mat2Quat(qr, data.site_xmat[sid_r])
    data.mocap_quat[mid_l] = ql
    data.mocap_quat[mid_r] = qr

    return ql, qr


def apply_configuration_to_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: mink.Configuration,
    mode: str = "qpos",  # "qpos" or "ctrl"
) -> None:
    """
    mode="qpos": actuator 무시하고 qpos를 직접 설정 (IK 확인용 가장 안전)
    mode="ctrl": nu개에 q를 넣음 (모델 actuator가 position servo일 때만)
    """
    if mode == "qpos":
        data.qpos[:] = configuration.q
    elif mode == "ctrl":
        if model.nu <= 0:
            data.qpos[:] = configuration.q
        else:
            n = min(model.nu, configuration.q.size)
            data.ctrl[:n] = configuration.q[:n]
    else:
        raise ValueError("mode must be 'qpos' or 'ctrl'")


# -------------------------
# Main loop (dual panda 스타일)
# -------------------------
def main():
    # 1) load model + add mocap targets if needed
    model = load_model_with_targets(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    configuration = mink.Configuration(model)

    # 2) pick EE sites
    ee_left, ee_right = pick_two_ee_sites(model)
    print("[INFO] EE sites:", ee_left, ee_right)

    # 3) tasks (dual panda 예시 스타일)
    left_task = mink.FrameTask(
        frame_name=ee_left,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.2,
        lm_damping=1.0,
    )
    right_task = mink.FrameTask(
        frame_name=ee_right,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.2,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=POSTURE_COST)

    tasks = [left_task, right_task, posture_task]

    # 4) viewer start
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # home keyframe 있으면 적용
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)

        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        # mocap targets = current EE
        tgt_quat_left, tgt_quat_right = initialize_mocap_targets_to_sites(model, data, ee_left, ee_right)
        mujoco.mj_forward(model, data)

        rate = RateLimiter(frequency=RATE_HZ, warn=False)

        # ✅ control apply mode:
        # - 먼저 qpos로 IK 동작 확인 추천
        APPLY_MODE = "qpos"   # "qpos" or "ctrl"

        while viewer.is_running():
            viewer.sync()

            # read target poses from mocap
            T_wt_left = mink.SE3.from_mocap_name(model, data, "target_left")
            T_wt_right = mink.SE3.from_mocap_name(model, data, "target_right")
            left_task.set_target(T_wt_left)
            right_task.set_target(T_wt_right)

            # IK iterations per cycle (dual panda 스타일)
            reached = False
            for _ in range(MAX_ITERS_PER_CYCLE):
                vel = mink.solve_ik(configuration, tasks, rate.dt, SOLVER, DAMPING)
                configuration.integrate_inplace(vel, rate.dt)

                # apply to mujoco
                apply_configuration_to_mujoco(model, data, configuration, mode=APPLY_MODE)

                mujoco.mj_step(model, data)

                # (optional) reached check using mocap target pos/quat
                # mocap quat은 data.mocap_quat에서 읽으면 됨
                tl_pos = data.mocap_pos[model.body("target_left").mocapid].copy()
                tr_pos = data.mocap_pos[model.body("target_right").mocapid].copy()
                tl_quat = data.mocap_quat[model.body("target_left").mocapid].copy()
                tr_quat = data.mocap_quat[model.body("target_right").mocapid].copy()

                reached = check_reached(
                    model, data, ee_left, ee_right,
                    tl_pos, tl_quat,
                    tr_pos, tr_quat,
                    POS_THRESHOLD, ORI_THRESHOLD,
                )
                viewer.sync()
                rate.sleep()
                if reached:
                    break

            # 안정화용 1스텝
            if not reached:
                mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()


if __name__ == "__main__":
    main()
