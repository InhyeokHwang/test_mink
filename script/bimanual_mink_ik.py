from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink


_XML = Path(__file__).parent.parent / "description" / "dual_arm" / "scene.xml"

SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 10

# target(red sphere) config
TARGET_RADIUS = 0.05
TARGET_RGBA = [1.0, 0.1, 0.1, 0.9]


def converge_ik_bimanual(
    configuration: mink.Configuration,
    tasks: dict,
    dt: float,
    solver: str,
    pos_threshold: float,
    ori_threshold: float,
    max_iters: int,
    ee_task_keys: Tuple[str, str] = ("eef_left", "eef_right"),
) -> bool:
    """두 EE task가 모두 threshold 안으로 들어오면 수렴으로 판단."""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks.values(), dt, solver, damping=1e-3)
        configuration.integrate_inplace(vel, dt)

        ok = True
        for k in ee_task_keys:
            err = tasks[k].compute_error(configuration)  # [pos(3), ori(3)]
            if np.linalg.norm(err[:3]) > pos_threshold or np.linalg.norm(err[3:]) > ori_threshold:
                ok = False
                break
        if ok:
            return True
    return False


def _ensure_mocap_target(spec: mujoco.MjSpec, target_body_name: str) -> None:
    """mocap body + 빨간 구 geom을 보장. 이미 있으면 body 재사용 + geom 추가(중복 가능)."""
    try:
        body = spec.body(target_body_name)
    except Exception:
        body = None

    if body is None:
        body = spec.worldbody.add_body(name=target_body_name, mocap=True)

    r = float(TARGET_RADIUS)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[r, r, r],
        rgba=TARGET_RGBA,
        contype=0,
        conaffinity=0,
    )


def load_model_with_two_targets(
    xml_path: Path,
    target_left: str = "target_left",
    target_right: str = "target_right",
) -> mujoco.MjModel:
    # MJCF(XML) -> MjSpec -> MjModel
    spec = mujoco.MjSpec.from_file(xml_path.as_posix())
    _ensure_mocap_target(spec, target_left)
    _ensure_mocap_target(spec, target_right)
    return spec.compile()


def reset_to_home_if_exists(model: mujoco.MjModel, data: mujoco.MjData, key_name: str = "home") -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)


def list_site_names(model: mujoco.MjModel) -> List[str]:
    names = []
    for i in range(model.nsite):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if nm:
            names.append(nm)
    return names


def pick_two_ee_sites(model: mujoco.MjModel) -> Tuple[str, str]:
    """
    양팔 EE site 이름 2개를 찾는다.
    우선순위:
      1) gripper_left / gripper_right
      2) gripper1 / gripper2
      3) 이름에 'gripper' 포함된 site 2개
      4) site 2개 이상이면 앞의 2개
      5) 그 외는 에러
    """
    # 1) 가장 명시적인 이름
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_left") != -1 and \
       mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_right") != -1:
        return "gripper_left", "gripper_right"

    # 2) 숫자 네이밍
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper1") != -1 and \
       mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper2") != -1:
        return "gripper1", "gripper2"

    # 3) 'gripper' 포함 site 2개
    sites = list_site_names(model)
    grippers = [s for s in sites if "gripper" in s.lower()]
    if len(grippers) >= 2:
        return grippers[0], grippers[1]

    # 4) 아무 site 2개
    if len(sites) >= 2:
        return sites[0], sites[1]

    raise RuntimeError(
        "Bimanual requires 2 EE sites, but model.nsite < 2.\n"
        "Add two sites like <site name='gripper_left' .../> and <site name='gripper_right' .../>"
    )


def main():
    # 1) 모델 로드 + mocap target 2개 보장
    model = load_model_with_two_targets(_XML, target_left="target_left", target_right="target_right")
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    # 2) EE site 두 개 선택
    ee_left, ee_right = pick_two_ee_sites(model)
    print("[INFO] Using EE sites:", ee_left, ee_right)

    # 3) Task 두 개 + posture
    eef_left_task = mink.FrameTask(
        frame_name=ee_left,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
        lm_damping=1.0,
    )
    eef_right_task = mink.FrameTask(
        frame_name=ee_right,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
        lm_damping=1.0,
    )

    posture_task = mink.PostureTask(model=model, cost=1e-2)

    tasks = {
        "eef_left": eef_left_task,
        "eef_right": eef_right_task,
        "posture": posture_task,
    }

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # 초기화
        reset_to_home_if_exists(model, data, "home")
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # mocap target을 각각 EE로 초기화 (1회)
        mink.move_mocap_to_frame(model, data, "target_left", ee_left, "site")
        mink.move_mocap_to_frame(model, data, "target_right", ee_right, "site")
        mujoco.mj_forward(model, data)

        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            dt = rate.dt
            viewer.sync()

            # 4) target 두 개의 목표 pose를 각각 set_target
            T_w_tl = mink.SE3.from_mocap_name(model, data, "target_left")
            T_w_tr = mink.SE3.from_mocap_name(model, data, "target_right")
            eef_left_task.set_target(T_w_tl)
            eef_right_task.set_target(T_w_tr)

            # 5) IK (두 EE 모두 수렴해야 True)
            converge_ik_bimanual(
                configuration,
                tasks,
                dt,
                SOLVER,
                POS_THRESHOLD,
                ORI_THRESHOLD,
                MAX_ITERS,
                ee_task_keys=("eef_left", "eef_right"),
            )

            # 6) 결과를 MuJoCo에 적용
            # NOTE: position actuator를 쓰는 모델이면, ctrl에 "목표 관절각"을 넣는 구조가 흔함.
            # 현재는 간단하게 앞쪽 nu개에 q를 넣는 방식(네 기존 코드 스타일)을 유지.
            if model.nu > 0:
                n = min(model.nu, configuration.q.size)
                data.ctrl[:n] = configuration.q[:n]
            else:
                data.qpos[:] = configuration.q

            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
