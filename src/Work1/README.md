# 实验二：三维坐标变换与 MVP 矩阵推导（GIF图在第七部分，需要等待片刻）
## 202411081008 冯丹蕊 计算机科学与技术（公费师范）
## 一、实验目标

1. 深入理解 3D 空间中的坐标变换流程（MVP 变换：Model-View-Projection）；
2. 独立推导并用代码实现模型变换、视图变换和透视投影变换矩阵；
3. 掌握 Taichi 框架的基本语法与矩阵操作，完成三维线框图形的渲染。

---

## 二、实验背景

在计算机图形学中，将三维世界中的物体显示到二维屏幕上，需要经历一条标准的坐标变换流水线：

$$\text{世界坐标} \xrightarrow{M_{model}} \text{相机坐标} \xrightarrow{M_{view}} \text{裁剪坐标} \xrightarrow{M_{proj}} \text{NDC} \xrightarrow{\text{视口变换}} \text{屏幕坐标}$$

本次实验在 Taichi 框架下手动推导并实现了这条流水线中的三个核心矩阵，并将初始三角形顶点：

- $v_0$: (2.0, 0.0, -2.0)
- $v_1$: (0.0, 2.0, -2.0)
- $v_2$: (-2.0, 0.0, -2.0)

经过完整 MVP 变换后映射至屏幕坐标，绘制为彩色线框三角形，并支持键盘交互旋转。

---

## 三、项目结构

```

CG-Lab/
├── src/
│   └── Work1/
│       ├── **init**.py
│       ├── main.py           # 必做：三角形 MVP 变换与线框渲染
│       └── main_cube.py      # 选做：3D 立方体姿态插值旋转
├── assets/
│   ├── triangle_demo.gif     # 必做效果演示
│   └── cube_demo.gif         # 选做效果演示
└── README.md

````

---

## 四、MVP 矩阵推导与实现

### 4.1 模型变换矩阵（Model Matrix）

模型变换负责对物体自身进行旋转、缩放、平移操作。本实验实现绕 Z 轴旋转角度 $\theta$ 的变换矩阵：

$$M_{model} = \begin{pmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

输入角度为角度制，需先转换为弧度：$\theta = \text{angle} \times \dfrac{\pi}{180}$

```python
@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c,   -s,  0.0, 0.0],
        [s,    c,  0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
````

### 4.2 视图变换矩阵（View Matrix）

视图变换将相机从世界坐标系的任意位置"移动"到原点，使其朝向 -Z 轴。本实验相机固定于 $(0, 0, 5)$，旋转方向已对齐，因此视图矩阵只需做平移：

$$M_{view} = \begin{pmatrix} 1 & 0 & 0 & -e_x \ 0 & 1 & 0 & -e_y \ 0 & 0 & 1 & -e_z \ 0 & 0 & 0 & 1 \end{pmatrix}$$

```python
@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
```

### 4.3 透视投影矩阵（Projection Matrix）

透视投影分两步完成：

---

**第一步：透视挤压 $M_{persp \to ortho}$**

将视锥体（Frustum）沿深度方向挤压为正交长方体，近截面保持不变：

$$M_{persp \to ortho} = \begin{pmatrix} n & 0 & 0 & 0 \ 0 & n & 0 & 0 \ 0 & 0 & n+f & -nf \ 0 & 0 & 1 & 0 \end{pmatrix}$$

> 注意 Z 轴符号约定：相机朝向 -Z 方向，因此 $n = -zNear$，$f = -zFar$。

---

**第二步：正交投影 $M_{ortho}$**

先平移使长方体中心与原点对齐，再缩放至 $[-1,1]^3$ 的 NDC 空间。

视锥体边界由视场角 $fov$ 推导：

$$t = \tan!\left(\frac{fov}{2}\right) \cdot |n|, \quad b = -t, \quad r = \text{aspect} \cdot t, \quad l = -r$$

$$M_{ortho} = \begin{pmatrix} \dfrac{2}{r-l} & 0 & 0 & -\dfrac{r+l}{r-l} [6pt] 0 & \dfrac{2}{t-b} & 0 & -\dfrac{t+b}{t-b} [6pt] 0 & 0 & \dfrac{2}{n-f} & -\dfrac{n+f}{n-f} [6pt] 0 & 0 & 0 & 1 \end{pmatrix}$$

**最终投影矩阵：**

$$M_{proj} = M_{ortho} \cdot M_{persp \to ortho}$$

```python
@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    M_p2o = ti.Matrix([
        [n,   0.0, 0.0,   0.0],
        [0.0, n,   0.0,   0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0,   0.0]
    ])
    M_ortho_scale = ti.Matrix([
        [2.0/(r-l), 0.0,       0.0,       0.0],
        [0.0,       2.0/(t-b), 0.0,       0.0],
        [0.0,       0.0,       2.0/(n-f), 0.0],
        [0.0,       0.0,       0.0,       1.0]
    ])
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r+l)/2.0],
        [0.0, 1.0, 0.0, -(t+b)/2.0],
        [0.0, 0.0, 1.0, -(n+f)/2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return M_ortho_scale @ M_ortho_trans @ M_p2o
```

---

## 五、完整变换流程

```python
@ti.kernel
def compute_transform(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view  = get_view_matrix(eye_pos)
    proj  = get_projection_matrix(45.0, 1.0, 0.1, 50.0)

    # MVP 右乘原则（列向量）
    mvp = proj @ view @ model

    for i in range(3):
        v4     = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        v_ndc  = v_clip / v_clip[3]               # 透视除法 → NDC [-1, 1]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0   # 视口变换 → [0, 1]
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0
```

每帧数据流如下：

```
顶点(世界坐标)
    ×M_model  →  旋转后坐标
    ×M_view   →  相机坐标系
    ×M_proj   →  裁剪坐标
    ÷w        →  NDC 空间 [-1, 1]
    视口变换  →  屏幕空间 [0, 1]
    GUI 绘制
```

---

## 六、运行方式

```bash
# 必做：三角形线框
uv run -m src.Work1.main

# 选做：3D 立方体插值旋转
uv run -m src.Work1.main_cube
```
<img width="803" height="121" alt="image" src="https://github.com/user-attachments/assets/88fbe9ad-2626-477e-9d0a-80c54516b441" />

### 交互说明

| 按键    | 效果        |
| ----- | --------- |
| `A`   | 逆时针旋转 10° |
| `D`   | 顺时针旋转 10° |
| `Esc` | 退出程序      |

---

## 七、效果演示

### 必做：彩色线框三角形
<img width="784" height="720" alt="work1_triangle_demo" src="https://github.com/user-attachments/assets/6c7270af-3729-4b8c-9048-7a20a53354d4" />

三条边分别以红、绿、蓝三色绘制，按 A/D 键可绕 Z 轴旋转，透视投影效果正确。

### 选做：3D 立方体姿态插值旋转

<img width="720" height="720" alt="work1_cube_demo" src="https://github.com/user-attachments/assets/695ca547-444d-4950-a879-0c997ab36c9e" />



立方体由 8 个顶点、12 条边构成，中心在原点，边长为 2。程序在绕 X 轴旋转（姿态1）与绕 Y 轴旋转（姿态2）之间进行矩阵线性插值，插值系数 $t$ 随角度正弦波动，实现两种姿态间的平滑过渡。

**插值核心逻辑：**

```python
M1 = get_model_matrix_x(angle)          # 姿态1：绕 X 轴
M2 = get_model_matrix_y(angle * 1.5)    # 姿态2：绕 Y 轴（不同步调）
M_model = (1.0 - interp_t) * M1 + interp_t * M2   # 线性插值
interp_t = (sin(angle * 0.05) + 1.0) / 2.0         # 系数在 [0,1] 正弦波动
```

> 说明：对旋转矩阵做线性插值（Matrix LERP）是近似方法，严格的旋转插值应使用球面线性插值（SLERP）或四元数插值。此处为教学目的采用矩阵 LERP 实现，效果上已能清晰展示姿态过渡。

---

## 八、实验总结

1. **MVP 变换流水线**：三个矩阵各司其职，顺序严格为 $M_{proj} \cdot M_{view} \cdot M_{model}$，不可颠倒，否则变换结果完全错误。

2. **Z 轴符号约定**：右手坐标系中相机朝向 -Z 方向，推导投影矩阵时必须令 $n = -zNear$、$f = -zFar$，忽略这一约定会导致图像翻转或消失。

3. **透视除法的本质**：MVP 变换后得到齐次坐标 $(x, y, z, w)$，除以 $w$ 才能还原为真实的 NDC 坐标，这一步是透视"近大远小"效果产生的根本原因。

4. **Taichi 函数层级**：`@ti.kernel` 是 GPU 入口，只能由 Python 侧调用；`@ti.func` 是设备端函数，只能在内核内部调用。二者不可混用，否则编译报错。

5. **旋转插值的局限性**：矩阵线性插值在插值系数接近 0.5 时会产生轻微的缩放失真，根本原因是旋转矩阵的插值中间态不一定仍是合法的旋转矩阵（行列式不为 1）。严格实现应使用四元数 SLERP。
