import taichi as ti
import math

# 初始化 Taichi，使用 CPU 后端（稳定）
ti.init(arch=ti.cpu)

# 立方体有 8 个顶点
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

# 立方体的 12 条边（顶点索引连接关系）
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 前面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 后面
    (0, 4), (1, 5), (2, 6), (3, 7)   # 连接前后
]

# ---------- 矩阵生成函数 ----------
@ti.func
def get_model_matrix_x(angle: ti.f32):
    """绕 X 轴旋转的模型矩阵（角度制）"""
    rad = angle * math.pi / 180.0
    c, s = ti.cos(rad), ti.sin(rad)
    return ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c,   -s,  0.0],
        [0.0, s,   c,   0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_model_matrix_y(angle: ti.f32):
    """绕 Y 轴旋转的模型矩阵（角度制）"""
    rad = angle * math.pi / 180.0
    c, s = ti.cos(rad), ti.sin(rad)
    return ti.Matrix([
        [c,   0.0, s,   0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0, c,   0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    """视图变换矩阵：将相机平移到原点"""
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """透视投影矩阵（含透视到正交 + 正交投影）"""
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    # 透视 -> 正交（挤压矩阵）
    M_persp_to_ortho = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # 正交投影矩阵（缩放到 [-1, 1]^3）
    M_ortho = ti.Matrix([
        [2.0/(r-l), 0.0, 0.0, -(r+l)/(r-l)],
        [0.0, 2.0/(t-b), 0.0, -(t+b)/(t-b)],
        [0.0, 0.0, 2.0/(n-f), -(n+f)/(n-f)],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    return M_ortho @ M_persp_to_ortho

# ---------- 核心变换计算 ----------
@ti.kernel
def compute_transform(angle: ti.f32, interp_t: ti.f32):
    """
    计算顶点变换并映射到屏幕坐标
    angle: 控制姿态1和姿态2的基础旋转角度
    interp_t: 0~1 之间的插值系数
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # 姿态1：绕 X 轴旋转 (角度 = angle)
    M1 = get_model_matrix_x(angle)
    
    # 姿态2：绕 Y 轴旋转 (角度 = angle * 1.5，产生不同步调)
    M2 = get_model_matrix_y(angle * 1.5)
    
    # 在两个姿态的模型矩阵之间进行线性插值（选做核心）
    M_model = (1.0 - interp_t) * M1 + interp_t * M2
    
    # 组合 MVP
    mvp = proj @ view @ M_model
    
    # 遍历 8 个顶点，进行变换与透视除法
    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]  # 透视除法，映射到 NDC
        
        # 视口变换：从 [-1,1] 映射到 [0,1] 的屏幕空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

# ---------- 主程序入口 ----------
def main():
    # 初始化立方体的 8 个顶点（中心在原点，边长 = 2）
    vertices[0] = [-1.0, -1.0, -1.0]
    vertices[1] = [ 1.0, -1.0, -1.0]
    vertices[2] = [ 1.0,  1.0, -1.0]
    vertices[3] = [-1.0,  1.0, -1.0]
    vertices[4] = [-1.0, -1.0,  1.0]
    vertices[5] = [ 1.0, -1.0,  1.0]
    vertices[6] = [ 1.0,  1.0,  1.0]
    vertices[7] = [-1.0,  1.0,  1.0]

    # 创建 GUI 窗口
    gui = ti.GUI("实验二选做：3D立方体姿态插值 (A/D旋转, Esc退出)", res=(700, 700))
    
    angle = 0.0  # 初始角度
    
    while gui.running:
        # 键盘事件监听
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 5.0   # 逆时针旋转
            elif gui.event.key == 'd':
                angle -= 5.0   # 顺时针旋转
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 插值系数 t 随角度正弦波动，在 0~1 之间平滑往返
        # 这样即使不按键，立方体也会在两种姿态间自动过渡
        interp_t = (ti.sin(angle * 0.05) + 1.0) / 2.0
        
        # 计算顶点变换
        compute_transform(angle, interp_t)
        
        # 绘制立方体的 12 条边（白色线框）
        for i, j in EDGES:
            gui.line(screen_coords[i], screen_coords[j], radius=2, color=0xFFFFFF)
        
        # 在窗口显示当前插值系数（可选效果）
        gui.text_content = f"插值系数 t = {interp_t:.2f}  |  角度 = {angle:.1f}°"
        
        # 刷新窗口
        gui.show()

if __name__ == "__main__":
    main()