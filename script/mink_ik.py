from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_XML = Path(__file__).parent.parent / "description" / "franka_emika_panda" / "mjx_scene.xml"


SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 10 # 크면 정확한데 느림, 작으면 빠르지만 덜 정확해짐

# target(red sphere) config
TARGET_RADIUS = 0.05
TARGET_RGBA = [1.0, 0.1, 0.1, 0.9]


def converge_ik(configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters):
    for _ in range(max_iters): 
        vel = mink.solve_ik(configuration, tasks.values(), dt, solver, damping=1e-3) 
        configuration.integrate_inplace(vel, dt)

        err = tasks["eef"].compute_error(configuration)
        if np.linalg.norm(err[:3]) <= pos_threshold and np.linalg.norm(err[3:]) <= ori_threshold:
            return True
    return False

def load_model_with_big_target(xml_path: Path, target_body_name: str = "target") -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml_path.as_posix())

    body = None
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
    return spec.compile()

def reset_to_home_if_exists(model: mujoco.MjModel, data: mujoco.MjData, key_name: str = "home") -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)


def pick_site_name(model: mujoco.MjModel) -> str:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper") != -1:
        return "gripper"
    if model.nsite > 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, 0)
    raise RuntimeError("No site exists in this model. Cannot run site-based FrameTask.")

def main():
    model = load_model_with_big_target(_XML)
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    ee_site = pick_site_name(model)
    print("[INFO] Using EE site:", ee_site)

    end_effector_task = mink.FrameTask(
        frame_name=ee_site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = {"eef": end_effector_task, "posture": posture_task}

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        reset_to_home_if_exists(model, data, "home")
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # target을 EE로 초기화 (한 번만)
        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)

        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            dt = rate.dt

            viewer.sync()

            # 지금 target mocap pose를 목표로 설정
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # IK
            converge_ik(configuration, tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)

            if model.nu > 0:
                n = min(model.nu, configuration.q.size)
                data.ctrl[:n] = configuration.q[:n]
            else:
                data.qpos[:] = configuration.q

            mujoco.mj_step(model, data)

            # 화면 갱신
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
