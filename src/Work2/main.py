import taichi as ti
import numpy as np

# 使用 GPU 后端（如果显卡兼容，否则改 ti.cpu）
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线总采样点数（B样条模式下会动态分配）

# ---------- 显存缓冲区 ----------
# 用于显示的颜色缓冲区（三个通道单独使用，以支持原子加）
pixels_r = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
pixels_g = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
pixels_b = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
# 合成用的向量场
display_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 曲线点缓冲区（固定大小，实际使用点数由 kernel 参数控制）
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# GUI 绘制用的控制点缓冲池（对象池）
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# ---------- 纯 Python 函数：De Casteljau 算法 ----------
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_pts = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_pts.append([x, y])
    return de_casteljau(next_pts, t)

# ---------- B样条曲线计算（均匀三次） ----------
def compute_bspline_points(control_points, total_samples):
    n = len(control_points)
    if n < 4:
        return np.empty((0, 2), dtype=np.float32)
    m = n - 3  # 分段数
    seg_samples = total_samples // m
    remainder = total_samples % m
    pts_list = []
    for seg in range(m):
        count = seg_samples + (1 if seg < remainder else 0)
        if count == 0:
            continue
        P0, P1, P2, P3 = control_points[seg:seg+4]
        for i in range(count):
            t = i / count  # 0 ~ 1
            u = 1.0 - t
            # 三次均匀B样条基函数
            B0 = (u * u * u) / 6.0
            B1 = (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0
            B2 = (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0
            B3 = (t * t * t) / 6.0
            x = B0 * P0[0] + B1 * P1[0] + B2 * P2[0] + B3 * P3[0]
            y = B0 * P0[1] + B1 * P1[1] + B2 * P2[1] + B3 * P3[1]
            pts_list.append([x, y])
    return np.array(pts_list, dtype=np.float32)

# ---------- Taichi Kernels ----------
@ti.kernel
def clear_pixels():
    for i, j in pixels_r:
        pixels_r[i, j] = 0.0
        pixels_g[i, j] = 0.0
        pixels_b[i, j] = 0.0

@ti.kernel
def composite():
    for i, j in display_pixels:
        display_pixels[i, j] = ti.Vector([pixels_r[i, j],
                                          pixels_g[i, j],
                                          pixels_b[i, j]])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    # 抗锯齿：对每个曲线点，影响其周围 3x3 像素，距离加权，原子累加
    for idx in range(n):
        pt = curve_points_field[idx]
        xf = pt[0] * WIDTH
        yf = pt[1] * HEIGHT

        ix = ti.cast(ti.floor(xf), ti.i32)
        iy = ti.cast(ti.floor(yf), ti.i32)

        # 遍历 3x3 邻域
        for dx in range(ix - 1, ix + 2):
            for dy in range(iy - 1, iy + 2):
                if 0 <= dx < WIDTH and 0 <= dy < HEIGHT:
                    # 像素中心坐标 (dx+0.5, dy+0.5)
                    cx = ti.cast(dx, ti.f32) + 0.5
                    cy = ti.cast(dy, ti.f32) + 0.5
                    dist = ti.sqrt((xf - cx) ** 2 + (yf - cy) ** 2)
                    # 衰减半径 1.5 像素
                    weight = ti.max(0.0, 1.0 - dist / 1.5)
                    if weight > 0.0:
                        # 原子累加绿色通道
                        ti.atomic_add(pixels_g[dx, dy], weight)
                        # 稍微带点蓝色使曲线呈青绿色，更美观（可选）
                        ti.atomic_add(pixels_b[dx, dy], weight * 0.2)

# ---------- 主程序 ----------
def main():
    window = ti.ui.Window("Bezier / B-Spline Curve (Press 'b' to switch)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []          # 当前控制点列表 (归一化坐标)
    mode = 'bezier'              # 'bezier' 或 'bspline'

    while window.running:
        # ---------- 事件处理 ----------
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added point {pos}, total={len(control_points)}")
            elif e.key == 'c':
                control_points = []
                print("Cleared all points.")
            elif e.key == 'b':
                mode = 'bspline' if mode == 'bezier' else 'bezier'
                print(f"Switched to {mode} mode.")
                window.title = f"Mode: {mode} (Press 'b' to switch)"

        # ---------- 清屏 ----------
        clear_pixels()

        current_count = len(control_points)
        curve_point_count = 0

        # ---------- 生成曲线 ----------
        if current_count >= 2:
            if mode == 'bezier':
                # 贝塞尔曲线
                curve_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
                for i in range(NUM_SEGMENTS + 1):
                    t = i / NUM_SEGMENTS
                    curve_np[i] = de_casteljau(control_points, t)
                curve_point_count = NUM_SEGMENTS + 1
                curve_points_field.from_numpy(curve_np)
            else:  # bspline
                if current_count >= 4:
                    curve_np = compute_bspline_points(control_points, NUM_SEGMENTS)
                    if curve_np.shape[0] > 0:
                        curve_point_count = curve_np.shape[0]
                        # 如果点数少于 field 大小，只填充前 curve_point_count 个
                        # 但 field 大小固定，我们直接 from_numpy 会要求大小完全匹配，所以需要先 resize 或切片
                        # 因为 field 大小是 NUM_SEGMENTS+1，而 curve_np 可能小于或等于，我们可以将 field 的前部分赋值为 curve_np
                        # 这里使用 from_numpy 要求 shape 完全一致，所以我们创建一个临时数组补全到指定长度
                        full_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
                        full_np[:curve_point_count] = curve_np
                        curve_points_field.from_numpy(full_np)
                    else:
                        curve_point_count = 0
                else:
                    curve_point_count = 0

        # ---------- 绘制曲线（GPU 并行） ----------
        if curve_point_count > 0:
            draw_curve_kernel(curve_point_count)

        # ---------- 合成并显示 ----------
        composite()
        canvas.set_image(display_pixels)

        # ---------- 绘制控制点和控制多边形 ----------
        if current_count > 0:
            # 控制点（红色圆点）
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))

            # 控制多边形（灰色连线）
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))

        # ---------- 显示 ----------
        window.show()

if __name__ == '__main__':
    main()