from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink


_HERE = Path(__file__).parent
_XML = _HERE.parent / "description" / "dual_arm" / "scene.xml"

SOLVER = "daqp"

# IK
POSTURE_COST = 1e-4
MAX_ITERS_PER_CYCLE = 20
DAMPING = 5e-4

# Convergence thresholds
POS_THRESHOLD = 1e-3
ORI_THRESHOLD = 1e-2

# Viewer loop rate
RATE_HZ = 100.0

# Mocap target
TARGET_RADIUS = 0.03
TARGET_RGBA_LEFT = [0.1, 0.9, 0.1, 0.9]
TARGET_RGBA_RIGHT = [0.1, 0.1, 0.9, 0.9]


def pick_two_ee_sites(model: mujoco.MjModel) -> Tuple[str, str]:
    common_pairs = [
        ("gripper_left", "gripper_right"),
    ]
    for a, b in common_pairs:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, a) != -1 and \
           mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, b) != -1:
            return a, b
    raise RuntimeError("EE sites not found: expected gripper_left/right")


def quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> float:
    return float(min(np.linalg.norm(q_current - q_target), np.linalg.norm(q_current + q_target)))


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = data.site_xpos[site_id].copy()
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return pos, quat


def check_reached(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_left_id: int,
    site_right_id: int,
    left_target_pos: np.ndarray,
    left_target_quat: np.ndarray,
    right_target_pos: np.ndarray,
    right_target_quat: np.ndarray,
    pos_threshold: float,
    ori_threshold: float,
) -> bool:
    meas_l_pos, meas_l_quat = site_pose(model, data, site_left_id)
    meas_r_pos, meas_r_quat = site_pose(model, data, site_right_id)

    err_pos_l = np.linalg.norm(meas_l_pos - left_target_pos)
    err_ori_l = quaternion_error(meas_l_quat, left_target_quat)
    err_pos_r = np.linalg.norm(meas_r_pos - right_target_pos)
    err_ori_r = quaternion_error(meas_r_quat, right_target_quat)

    return (
        err_pos_l <= pos_threshold
        and err_ori_l <= ori_threshold
        and err_pos_r <= pos_threshold
        and err_ori_r <= ori_threshold
    )


def _ensure_mocap_target(spec: mujoco.MjSpec, name: str, rgba: List[float]) -> None:
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


def load_model(xml_path: Path) -> mujoco.MjModel:
    try:
        spec = mujoco.MjSpec.from_file(xml_path.as_posix())
        _ensure_mocap_target(spec, "target_left", TARGET_RGBA_LEFT)
        _ensure_mocap_target(spec, "target_right", TARGET_RGBA_RIGHT)
        return spec.compile()
    except Exception as e:
        print(f"[WARN] MjSpec injection failed ({type(e).__name__}: {e}). "
              f"Falling back to from_xml_path; assuming targets already exist in XML.")
        return mujoco.MjModel.from_xml_path(xml_path.as_posix())


def initialize_model() -> Tuple[mujoco.MjModel, mujoco.MjData, mink.Configuration]:
    model = load_model(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    configuration = mink.Configuration(model)
    return model, data, configuration


def _actuator_joint_id(model: mujoco.MjModel, act_id: int) -> Optional[int]:
    try:
        trnid = model.actuator_trnid[act_id]
        j_id = int(trnid[0])
        if 0 <= j_id < model.njnt:
            return j_id
    except Exception:
        pass
    return None


def build_ctrl_map_for_joints(model: mujoco.MjModel) -> Dict[int, int]:
    m: Dict[int, int] = {}
    if model.nu <= 0:
        return m
    for a in range(model.nu):
        j = _actuator_joint_id(model, a)
        if j is None:
            continue
        if j not in m:
            m[j] = a
    return m


def apply_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: mink.Configuration,
    joint2act: Dict[int, int],
) -> None:
    if model.nu <= 0 or not joint2act:
        data.qpos[:] = configuration.q
        return

    for j_id, a_id in joint2act.items():
        qadr = int(model.jnt_qposadr[j_id])
        jtype = int(model.jnt_type[j_id])
        if jtype in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        data.ctrl[a_id] = float(configuration.q[qadr])


def initialize_mocap_targets_to_sites(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_left_name: str,
    site_right_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    site_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_left_name)
    site_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_right_name)
    if site_left_id < 0 or site_right_id < 0:
        raise RuntimeError("EE sites not found (check your site names).")

    mocap_l = model.body("target_left").mocapid
    mocap_r = model.body("target_right").mocapid

    data.mocap_pos[mocap_l] = data.site_xpos[site_left_id].copy()
    data.mocap_pos[mocap_r] = data.site_xpos[site_right_id].copy()

    ql = np.empty(4, dtype=np.float64)
    qr = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(ql, data.site_xmat[site_left_id])
    mujoco.mju_mat2Quat(qr, data.site_xmat[site_right_id])
    data.mocap_quat[mocap_l] = ql
    data.mocap_quat[mocap_r] = qr

    return ql, qr


def main():
    model, data, configuration = initialize_model()

    # initial pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    # EE sites
    ee_left, ee_right = pick_two_ee_sites(model)
    print(f"[INFO] EE sites: {ee_left}, {ee_right}")

    site_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_left)
    site_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_right)

    # tasks
    left_task = mink.FrameTask(
        frame_name=ee_left, frame_type="site",
        position_cost=1.0, orientation_cost=0.2,
        lm_damping=1.0,
    )
    right_task = mink.FrameTask(
        frame_name=ee_right, frame_type="site",
        position_cost=1.0, orientation_cost=0.2,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=POSTURE_COST)
    tasks = [left_task, right_task, posture_task]
    posture_task.set_target_from_configuration(configuration)

    # ctrl map
    joint2act = build_ctrl_map_for_joints(model)

    # init mocap targets to current EE
    initialize_mocap_targets_to_sites(model, data, ee_left, ee_right)
    mujoco.mj_forward(model, data)

    rate = RateLimiter(frequency=RATE_HZ, warn=False)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            frame_dt = rate.dt
            ik_dt = frame_dt / MAX_ITERS_PER_CYCLE

            # mocap -> task target
            T_wt_left = mink.SE3.from_mocap_name(model, data, "target_left")
            T_wt_right = mink.SE3.from_mocap_name(model, data, "target_right")
            left_task.set_target(T_wt_left)
            right_task.set_target(T_wt_right)

            mocap_l = model.body("target_left").mocapid
            mocap_r = model.body("target_right").mocapid
            left_target_pos = data.mocap_pos[mocap_l].copy()
            right_target_pos = data.mocap_pos[mocap_r].copy()
            left_target_quat = data.mocap_quat[mocap_l].copy()
            right_target_quat = data.mocap_quat[mocap_r].copy()

            # IK sub-iterations
            reached = False
            for _ in range(MAX_ITERS_PER_CYCLE):
                vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, DAMPING)
                configuration.integrate_inplace(vel, ik_dt)

                apply_configuration(model, data, configuration, joint2act=joint2act)
                mujoco.mj_step(model, data)

                reached = check_reached(
                    model, data,
                    site_left_id, site_right_id,
                    left_target_pos, left_target_quat,
                    right_target_pos, right_target_quat,
                    POS_THRESHOLD, ORI_THRESHOLD,
                )
                if reached:
                    break

            # render/sleep exactly once per frame
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()