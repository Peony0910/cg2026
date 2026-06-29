import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# ---------- UI 参数 ----------
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
samples_per_pixel = ti.field(ti.i32, shape=())
use_glass = ti.field(ti.i32, shape=())  # 0=红色漫反射，1=玻璃

# ---------- 材质常量 ----------
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

# ---------- 工具函数 ----------
@ti.func
def normalize(v):
    return v / (v.norm() + 1e-8)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    cos_theta = -N.dot(I)
    N_local = N
    eta_local = eta  # 默认值，后面可能被覆盖
    if cos_theta < 0.0:
        N_local = -N
        cos_theta = -N_local.dot(I)
        eta_local = 1.0 / eta
    sin_theta2 = eta_local * eta_local * (1.0 - cos_theta * cos_theta)
    result = ti.Vector([0.0, 0.0, 0.0])
    if sin_theta2 <= 1.0:
        cos_phi = ti.sqrt(1.0 - sin_theta2)
        R = eta_local * I + (eta_local * cos_theta - cos_phi) * N_local
        result = normalize(R)
    return result

# ---------- 几何求交 ----------
@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 左侧球体（动态切换）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        if use_glass[None] == 1:
            hit_c = ti.Vector([1.0, 1.0, 1.0])   # 玻璃
            hit_mat = MAT_GLASS
        else:
            hit_c = ti.Vector([0.8, 0.1, 0.1])   # 红色漫反射
            hit_mat = MAT_DIFFUSE

    # 右侧镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 地板（棋盘格）
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        ix_int = ti.cast(ix, ti.i32)
        iz_int = ti.cast(iz, ti.i32)
        if (ix_int + iz_int) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def shadow_ray(ro, rd, t_max):
    t, _, _, _ = scene_intersect(ro, rd)
    return 0 < t < t_max

# ---------- 主渲染核 ----------
@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    n_samples = samples_per_pixel[None]
    if n_samples < 1:
        n_samples = 1

    for i, j in pixels:
        color_sum = ti.Vector([0.0, 0.0, 0.0])
        for _ in range(n_samples):
            dx = ti.random(ti.f32) - 0.5
            dy = ti.random(ti.f32) - 0.5
            u = (i + dx - res_x / 2.0) / res_y * 2.0
            v = (j + dy - res_y / 2.0) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            final_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])

            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)
                if t > 1e9:
                    final_color += throughput * bg_color
                    break

                p = ro + rd * t

                if mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = reflect(rd, N)
                    throughput *= 0.8 * obj_color
                    continue

                elif mat_id == MAT_GLASS:
                    eta = 1.5
                    refl_dir = reflect(rd, N)
                    refr_dir = refract(rd, N, eta)
                    if refr_dir.norm() < 1e-6:  # 全反射
                        ro = p + N * 1e-4
                        rd = refl_dir
                        throughput *= 0.8 * obj_color
                    else:
                        cos_theta = ti.abs(N.dot(rd))
                        fresnel = 0.04 + 0.96 * ti.pow(1.0 - cos_theta, 5.0)
                        if ti.random(ti.f32) < fresnel:
                            ro = p + N * 1e-4
                            rd = refl_dir
                        else:
                            ro = p - N * 1e-4
                            rd = refr_dir
                        throughput *= 0.9 * obj_color
                    continue

                else:  # MAT_DIFFUSE
                    L = normalize(light_pos - p)
                    shadow_orig = p + N * 1e-4
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = shadow_ray(shadow_orig, L, dist_to_light)

                    ambient = 0.2 * obj_color
                    direct_light = ambient
                    if not in_shadow:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse

                    final_color += throughput * direct_light
                    break

            color_sum += final_color

        pixels[i, j] = ti.math.clamp(color_sum / n_samples, 0.0, 1.0)

# ---------- 主函数 ----------
def main():
    window = ti.ui.Window("Ray Tracing (必做红球+镜面 | 选做玻璃+MSAA)", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3
    samples_per_pixel[None] = 4
    use_glass[None] = 0

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.70, 0.05, 0.28, 0.32):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
            samples_per_pixel[None] = gui.slider_int('MSAA Samples (选做)', samples_per_pixel[None], 1, 16)

            if gui.button("Switch to Glass (Bonus)" if use_glass[None] == 0 else "Switch to Diffuse (Base)"):
                use_glass[None] = 1 - use_glass[None]

        window.show()

if __name__ == '__main__':
    main()