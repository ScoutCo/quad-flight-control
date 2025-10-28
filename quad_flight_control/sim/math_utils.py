import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_vector(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return v * 0.0
    return v / norm


def clamp(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(value, lower), upper)


def skew(vec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ]
    )


def _ensure_quat_array(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float).reshape(4)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    # Project convention lists scalar-first (w, x, y, z) to match ArduPilot / common aerospace tools,
    # while SciPy expects scalar-last (x, y, z, w).
    q = _ensure_quat_array(q)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    # Helper to bring SciPy's [x, y, z, w] output back to the project convention.
    q = _ensure_quat_array(q)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def _rotation_from_wxyz(q: np.ndarray) -> R:
    # ensure every SciPy Rotation sees normalized XYZW quaternions.
    return R.from_quat(_quat_wxyz_to_xyzw(quat_normalize(_ensure_quat_array(q))))


def _quat_wxyz_from_rotation(rotation: R) -> np.ndarray:
    # Normalize the result so downstream consumers always get unit quaternions in WXYZ order.
    return quat_normalize(_quat_xyzw_to_wxyz(rotation.as_quat()))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return normalize_vector(q)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    rotation = _rotation_from_wxyz(q1) * _rotation_from_wxyz(q2)
    return _quat_wxyz_from_rotation(rotation)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    return _rotation_from_wxyz(q).as_matrix()


def rotation_matrix_to_quat(r: np.ndarray) -> np.ndarray:
    return _quat_wxyz_from_rotation(R.from_matrix(r))


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize_vector(axis)
    return _quat_wxyz_from_rotation(R.from_rotvec(axis * angle))


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Return roll/pitch/yaw using the aerospace Z-Y-X convention."""

    # SciPy returns angles in the order of the axes we request. We ask for the
    # conventional aerospace sequence (yaw Z, pitch Y, roll X) and then reorder
    # the components so callers continue to receive roll, pitch, yaw.
    yaw, pitch, roll = _rotation_from_wxyz(q).as_euler("zyx")
    return np.array([roll, pitch, yaw], dtype=float)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""

    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return quat_normalize(result)

    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    if sin_theta_0 <= 1e-6:
        return q1.copy()

    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return quat_normalize(result)


def integrate_quaternion(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    delta_rotation = R.from_rotvec(np.asarray(omega_body, dtype=float) * dt)
    updated_rotation = _rotation_from_wxyz(q) * delta_rotation
    return _quat_wxyz_from_rotation(updated_rotation)


def quaternion_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    rotation_error = _rotation_from_wxyz(q_target) * _rotation_from_wxyz(q_current).inv()
    q_err = _quat_wxyz_from_rotation(rotation_error)
    if q_err[0] < 0.0:
        q_err *= -1.0
    return q_err


def quaternion_error_to_rotation_vector(q_err: np.ndarray) -> np.ndarray:
    return R.from_quat(_quat_wxyz_to_xyzw(quat_normalize(q_err))).as_rotvec()
