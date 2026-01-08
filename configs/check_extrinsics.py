import json
import numpy as np

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Cuaternión nulo")
    return q / n

def quat_to_R_xyzw(q):
    # q = [x, y, z, w]
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=float)
    return R

def check_rotation(R, tol_ortho=1e-6, tol_det=1e-6):
    I = np.eye(3)
    ortho_err = np.linalg.norm(R.T @ R - I, ord='fro')
    det = np.linalg.det(R)
    det_err = abs(det - 1.0)
    ok = (ortho_err < tol_ortho) and (det_err < tol_det)
    return ok, det, ortho_err

def axis_angle_from_quat_xyzw(q):
    # Devuelve (axis, angle_rad). Maneja casos cerca de 0 y de pi.
    x, y, z, w = quat_normalize(q)
    w_clamped = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * np.arccos(w_clamped)  # [0, 2pi]
    s = np.sqrt(max(0.0, 1.0 - w_clamped*w_clamped))  # = |sin(angle/2)|

    if s < 1e-12:
        # ángulo ~ 0: eje arbitrario; devolvemos el más estable
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([x, y, z]) / s

    # Normalización final por estabilidad numérica
    axis = axis / np.linalg.norm(axis)
    return axis, float(angle)

def analyze_extrinsics(extrinsics_dict, expect_pi_about_x_sensors=("lidar", "radar")):
    for name, data in extrinsics_dict.items():
        q = data["quaternion"]
        t = np.asarray(data["translation"], dtype=float)

        R = quat_to_R_xyzw(q)
        okR, det, ortho_err = check_rotation(R)

        axis, ang = axis_angle_from_quat_xyzw(q)

        # Métricas simples para el "≈ pi y eje ~ x"
        ang_to_pi = abs(ang - np.pi)
        axis_to_x = np.linalg.norm(axis - np.array([1.0, 0.0, 0.0]))
        axis_to_minus_x = np.linalg.norm(axis - np.array([-1.0, 0.0, 0.0]))
        axis_to_x_best = min(axis_to_x, axis_to_minus_x)

        tag_expect = any(k in name.lower() for k in expect_pi_about_x_sensors)

        print(f"\n=== {name} ===")
        print(f"t = {t}")
        print(f"det(R) = {det:.12f} | ||R^T R - I||_F = {ortho_err:.3e} | OK={okR}")
        print(f"axis = {axis} | angle = {ang:.6f} rad ({ang*180/np.pi:.3f} deg)")
        if tag_expect:
            print(f"[expect ~pi about x] | |angle-pi| = {ang_to_pi:.6f} rad | dist(axis, ±x) = {axis_to_x_best:.6f}")


# ---- Ejemplo de uso ----
extrinsics = json.loads(open("/home/gondeleon/Full_v5 - GPS-AHRS-IMU/configs/extrinsics.json","r").read())
analyze_extrinsics(extrinsics)