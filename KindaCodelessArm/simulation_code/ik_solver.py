import math
import numpy as np
import mujoco


# Joint names used for IK (gripper excluded — it's not part of positioning)
IK_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

ALL_JOINT_NAMES = IK_JOINT_NAMES + ["gripper"]


def solve_ik(
    m,
    d,
    target_xyz: list[float],
    max_iterations: int = 200,
    tolerance: float = 0.001,
    step_size: float = 0.5,
    initial_angles_deg: dict[str, float] = None,
) -> dict[str, float]:
    """
    Jacobian-based IK: finds joint angles that place gripperframe at target_xyz.

    Uses a damped least-squares (Levenberg-Marquardt) approach on the translational
    Jacobian to iteratively move the end-effector site toward the target.

    Returns joint angles in degrees for all 6 joints (gripper held at 0).
    """
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in IK_JOINT_NAMES]
    dof_ids = [m.jnt_dofadr[jid] for jid in joint_ids]
    n_dof = len(dof_ids)

    # Work on a copy so we don't disturb the live simulation state
    d_ik = mujoco.MjData(m)
    d_ik.qpos[:] = d.qpos[:]
    d_ik.qvel[:] = d.qvel[:]

    # Apply initial guess if provided
    if initial_angles_deg is not None:
        for jname, jid in zip(IK_JOINT_NAMES, joint_ids):
            if jname in initial_angles_deg:
                d_ik.qpos[m.jnt_qposadr[jid]] = math.radians(initial_angles_deg[jname])

    target = np.array(target_xyz, dtype=np.float64)
    damping = 1e-4

    for iteration in range(max_iterations):
        mujoco.mj_forward(m, d_ik)

        # Current end-effector position
        ee_pos = d_ik.site_xpos[site_id].copy()
        error = target - ee_pos
        dist = np.linalg.norm(error)

        if dist < tolerance:
            break

        # Compute full Jacobian, then extract columns for our DOFs
        jacp = np.zeros((3, m.nv), dtype=np.float64)
        mujoco.mj_jacSite(m, d_ik, jacp, None, site_id)
        J = jacp[:, dof_ids]  # 3 x n_dof

        # Damped least-squares: dq = J^T (J J^T + lambda I)^-1 error
        JJT = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)
        dq *= step_size

        # Apply joint updates, respecting limits
        for i, jid in enumerate(joint_ids):
            qadr = m.jnt_qposadr[jid]
            d_ik.qpos[qadr] += dq[i]
            # Clamp to joint limits
            if m.jnt_limited[jid]:
                lo = m.jnt_range[jid, 0]
                hi = m.jnt_range[jid, 1]
                d_ik.qpos[qadr] = np.clip(d_ik.qpos[qadr], lo, hi)

    # Read final joint angles
    result_deg = {}
    for jname, jid in zip(IK_JOINT_NAMES, joint_ids):
        qadr = m.jnt_qposadr[jid]
        result_deg[jname] = math.degrees(d_ik.qpos[qadr])
    result_deg["gripper"] = 0.0

    return result_deg
