import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 材质参数
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())
# 切换光照模型：0=Phong, 1=Blinn-Phong
use_blinn = ti.field(ti.i32, shape=())

@ti.func
def normalize(v):
    return v / (v.norm() + 1e-8)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# ---------- 几何体求交 ----------
@ti.func
def intersect_sphere(ro, rd, center, radius):
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    H = apex.y - base_y
    k = (radius / H) ** 2
    ro_local = ro - apex
    A = rd.x*rd.x + rd.z*rd.z - k * rd.y*rd.y
    B = 2.0 * (ro_local.x*rd.x + ro_local.z*rd.z - k * ro_local.y*rd.y)
    C = ro_local.x*ro_local.x + ro_local.z*ro_local.z - k * ro_local.y*ro_local.y
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    if ti.abs(A) > 1e-5:
        delta = B*B - 4.0*A*C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0*A)
            t2 = (-B + ti.sqrt(delta)) / (2.0*A)
            if t1 > t2:
                t1, t2 = t2, t1
            y1 = ro_local.y + t1 * rd.y
            if t1 > 0 and -H <= y1 <= 0:
                t = t1
            else:
                y2 = ro_local.y + t2 * rd.y
                if t2 > 0 and -H <= y2 <= 0:
                    t = t2
            if t > 0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
    return t, normal

# ---------- 阴影射线（选做） ----------
@ti.func
def shadow_ray(ro, rd, t_max):
    # 检测从 ro 沿 rd 方向在 t_max 范围内是否有遮挡
    hit = False
    t_sph, _ = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
    if 0 < t_sph < t_max:
        hit = True
    if not hit:
        t_cone, _ = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < t_max:
            hit = True
    return hit

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        hit_pos = ti.Vector([0.0, 0.0, 0.0])

        # 球体
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            hit_pos = ro + rd * min_t

        # 圆锥
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])
            hit_pos = ro + rd * min_t

        color = ti.Vector([0.05, 0.15, 0.15])

        if min_t < 1e9:
            N = hit_normal
            P = hit_pos
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])

            L = normalize(light_pos - P)
            V = normalize(ro - P)

            # 硬阴影
            in_shadow = False
            shadow_origin = P + N * 1e-4
            shadow_dir = L
            shadow_t_max = (light_pos - shadow_origin).norm()
            if shadow_ray(shadow_origin, shadow_dir, shadow_t_max):
                in_shadow = True

            ambient = Ka[None] * light_color * hit_color

            # 初始化漫反射和高光
            diffuse = ti.Vector([0.0, 0.0, 0.0])
            specular = ti.Vector([0.0, 0.0, 0.0])

            if not in_shadow:
                diff = ti.max(0.0, N.dot(L))
                diffuse = Kd[None] * diff * light_color * hit_color

                # 高光计算：提前声明 spec_val
                spec_val = 0.0
                if use_blinn[None] == 1:
                    H = normalize(L + V)
                    spec_val = ti.max(0.0, N.dot(H)) ** shininess[None]
                else:
                    R = normalize(reflect(-L, N))
                    spec_val = ti.max(0.0, R.dot(V)) ** shininess[None]
                specular = Ks[None] * spec_val * light_color

            color = ambient + diffuse + specular

        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Phong / Blinn-Phong with Shadows", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    # 初始化参数
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0
    use_blinn[None] = 0

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.25):
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)
            # 切换按钮
            if gui.button("Switch to Blinn-Phong" if use_blinn[None] == 0 else "Switch to Phong"):
                use_blinn[None] = 1 - use_blinn[None]

        window.show()

if __name__ == '__main__':
    main()