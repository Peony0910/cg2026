import taichi as ti

ti.init(arch=ti.gpu)

# ---------- 物理参数 ----------
N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0
k_shear = 5000.0
k_bend = 2000.0
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

# 碰撞球体（选做）——改为场，以便传入 scene.particles
sphere_center = ti.Vector.field(3, dtype=float, shape=1)
sphere_radius = 0.4

# ---------- 数据场 ----------
x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

max_springs = N * N * 6
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

# ---------- 初始化 ----------
@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        # 结构弹簧
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()

        # 剪切弹簧
        if i < N - 1 and j < N - 1:
            idx_diag = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
        if i < N - 1 and j > 0:
            idx_other = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_other])
            spring_lengths[c] = (x[idx] - x[idx_other]).norm()

        # 弯曲弹簧
        if i < N - 2:
            idx_next2 = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_next2])
            spring_lengths[c] = (x[idx] - x[idx_next2]).norm()
        if j < N - 2:
            idx_down2 = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down2])
            spring_lengths[c] = (x[idx] - x[idx_down2]).norm()

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()
    # 初始化球体中心（只做一次）
    if sphere_center[0].norm() == 0.0:
        sphere_center[0] = ti.Vector([0.0, -0.5, 0.0])

# ---------- 力计算（含碰撞） ----------
@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    # 重力 + 阻尼
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]

    # 弹簧力（结构、剪切、弯曲）
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -k_s * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

    # ---- 球体碰撞（选做） ----
    center = sphere_center[0]
    for i in range(N * N):
        if is_fixed[i] == 0:
            p = pos[i]
            dir_to_center = p - center
            dist = dir_to_center.norm()
            if dist < sphere_radius and dist > 1e-8:
                normal = dir_to_center / dist
                penetration = sphere_radius - dist
                pos[i] += normal * penetration
                vn = vel[i].dot(normal)
                if vn < 0:
                    vel[i] -= (1 + 0.3) * vn * normal

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vn = vel[idx].norm()
    if vn > max_velocity:
        vel[idx] = vel[idx] / vn * max_velocity

# ---------- 积分核 ----------
@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

# ---------- 主循环 ----------
def main():
    init_cloth()
    window = ti.ui.Window("Mass Spring System (Bonus: Shear+Bend+Sphere)", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1
    paused = False
    show_collision = True

    while window.running:
        # GUI
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.42)
        window.GUI.text("Integration Method:")
        for method, name in enumerate(["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]):
            prefix = "[*] " if current_method == method else "[ ] "
            if window.GUI.button(prefix + name):
                current_method = method
                init_cloth()
        window.GUI.text("")
        if window.GUI.button("Pause" if not paused else "Resume"):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()
        window.GUI.text("")
        window.GUI.text("Bonus Features:")
        if window.GUI.button("Toggle Collision Sphere"):
            show_collision = not show_collision
        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        # 渲染
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        if show_collision:
            scene.particles(sphere_center, radius=sphere_radius, color=(1.0, 0.3, 0.3))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()