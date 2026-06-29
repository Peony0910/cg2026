# 实验三：贝塞尔曲线与光栅化基础（GIF图在第七部分，需要稍等片刻加载）

**202411081008-冯丹蕊-计算机科学与技术（公费师范）**

---

## 一、实验目标

1. 理解贝塞尔曲线（Bézier Curve）的几何意义与数学本质；
2. 独立推导并用代码实现 De Casteljau 递归插值算法；
3. 掌握光栅化基础概念：在像素缓冲区（Frame Buffer）中直接操作像素；
4. 掌握现代图形界面中鼠标点击与键盘交互事件的处理方式；
5. 选做：实现基于距离衰减的反走样渲染与均匀三次 B 样条曲线绘制。

---

## 二、数学原理

### 2.1 De Casteljau 算法

贝塞尔曲线由一组控制点 $P_0, P_1, \dots, P_{n-1}$ 决定。引入参数 $t \in [0, 1]$，当 $t$ 从 0 连续变化到 1 时，所有插值点连在一起即构成整条曲线。

De Casteljau 算法通过递归线性插值求取曲线上的点：

**第一层插值：** 对相邻两点 $P_i$、$P_{i+1}$，在比例 $t$ 处插值：

$$P_i^{(1)} = (1-t)\,P_i + t\,P_{i+1}$$

**递归：** 对插值出的 $n-1$ 个新点重复上述操作，直至只剩一个点。

**终止：** 最终唯一的点即为曲线在参数 $t$ 处的精确位置。

以三个控制点 $P_0, P_1, P_2$ 为例，完整推导如下：

$$P_0^{(1)} = (1-t)P_0 + tP_1, \quad P_1^{(1)} = (1-t)P_1 + tP_2$$

$$P_0^{(2)} = (1-t)P_0^{(1)} + tP_1^{(1)} = (1-t)^2 P_0 + 2t(1-t)P_1 + t^2 P_2$$

这正是二次贝塞尔曲线的伯恩斯坦展开式，验证了 De Casteljau 算法的正确性。

代码实现（纯 Python 递归）：

```python
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
````

### 2.2 均匀三次 B 样条曲线（选做）

贝塞尔曲线存在两个局限：一是全局控制性（任意控制点改变影响整条曲线），二是阶数与控制点数绑定（控制点多时计算量急剧增大）。

B 样条曲线通过分段多项式基函数解决了这两个问题，实现了**局部控制**。

对于均匀三次 B 样条，每 4 个相邻控制点 $P_0, P_1, P_2, P_3$ 构成一段曲线，基函数为：

$$B_0(t) = \frac{(1-t)^3}{6}, \quad B_1(t) = \frac{3t^3 - 6t^2 + 4}{6}$$

$$B_2(t) = \frac{-3t^3 + 3t^2 + 3t + 1}{6}, \quad B_3(t) = \frac{t^3}{6}$$

该段曲线上参数 $t$ 处的点为：

$$P(t) = B_0(t),P_0 + B_1(t),P_1 + B_2(t),P_2 + B_3(t),P_3$$

$n$ 个控制点（$n \ge 4$）可生成 $n-3$ 段曲线平滑拼接的完整 B 样条。

---

## 三、光栅化原理

光栅化是将连续的几何坐标映射到离散像素网格的过程。本实验的光栅化流程如下：

1. **帧缓冲区**：创建 $800 \times 800$ 的 Taichi Field（`pixels_r/g/b`），模拟屏幕显存，每个像素存储 RGB 三个独立浮点通道。
2. **坐标映射**：De Casteljau 算法输出 $[0,1]$ 归一化浮点坐标，乘以屏幕宽高得到像素坐标：
   $$ix = \lfloor x_f \cdot W \rfloor, \quad iy = \lfloor y_f \cdot H \rfloor$$
3. **写入像素**：将对应索引处的颜色通道赋值，即"点亮像素"。
4. **反走样（选做）**：不直接截断到单一像素，而是考察 $3 \times 3$ 邻域，按距离衰减权重通过原子加（`ti.atomic_add`）叠加贡献，消除阶梯状锯齿。

**CPU-GPU 数据流（Batching）：**

```
CPU 侧（Python）              GPU 侧（Taichi Kernel）
────────────────────         ──────────────────────────
de_casteljau × 1001 次   →   from_numpy 一次性传输
所有采样点存入 NumPy 数组  →   draw_curve_kernel 并行点亮像素
```

直接在 Python 循环中逐点写 GPU Field 会产生大量 PCIe 通信开销，Batching 是解决帧率卡顿的关键。

---

## 四、项目结构

```
CG-Lab/
├── src/
│   └── Work2/
│       ├── __init__.py
│       └── main.py           # 必做 + 选做一体（按 b 键切换模式）
├── assets/
│   ├── bezier_demo.gif       # 必做：贝塞尔曲线交互演示
│   └── bspline_demo.gif      # 选做：B 样条模式切换演示
└── README.md
```

---

## 五、核心代码说明

### 5.1 显存缓冲区设计

```python
# 三通道分离存储，支持 ti.atomic_add 原子累加（反走样所需）
pixels_r = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
pixels_g = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
pixels_b = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 合成后的显示用 Field
display_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 曲线点缓冲区（固定大小，避免动态申请显存）
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# 控制点对象池（固定大小 100，多余位置藏到屏幕外 -10.0）
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
```

三通道分离的设计原因：Taichi 的 `ti.atomic_add` 只能作用于标量 Field，反走样的多像素叠加必须对每个通道单独原子加，最后由 `composite()` 内核合成为向量场再送入显示。

### 5.2 反走样 GPU 内核（选做）

```python
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for idx in range(n):
        pt = curve_points_field[idx]
        xf = pt[0] * WIDTH
        yf = pt[1] * HEIGHT
        ix = ti.cast(ti.floor(xf), ti.i32)
        iy = ti.cast(ti.floor(yf), ti.i32)
        # 遍历 3×3 邻域，按距离衰减加权
        for dx in range(ix - 1, ix + 2):
            for dy in range(iy - 1, iy + 2):
                if 0 <= dx < WIDTH and 0 <= dy < HEIGHT:
                    cx = ti.cast(dx, ti.f32) + 0.5
                    cy = ti.cast(dy, ti.f32) + 0.5
                    dist = ti.sqrt((xf - cx) ** 2 + (yf - cy) ** 2)
                    weight = ti.max(0.0, 1.0 - dist / 1.5)  # 衰减半径 1.5px
                    if weight > 0.0:
                        ti.atomic_add(pixels_g[dx, dy], weight)
                        ti.atomic_add(pixels_b[dx, dy], weight * 0.2)
```

`@ti.kernel` 中的外层 `for idx in range(n)` 由 GPU 自动并行展开，1001 个曲线点同时计算，每个点影响最多 9 个邻域像素，`ti.atomic_add` 保证多线程并发写入同一像素时不产生竞争条件。

### 5.3 控制点对象池

由于 `canvas.circles()` 只接受固定长度的 Taichi Field，控制点数量动态变化，直接传可变长度数组会报错。解决方法：

```python
# 创建全填充为 -10.0 的 NumPy 数组（-10.0 在屏幕范围 [0,1] 之外，不可见）
np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
# 只把真实控制点覆盖到前 current_count 个位置
np_points[:current_count] = np.array(control_points, dtype=np.float32)
# 一次性传入固定大小的 Field
gui_points.from_numpy(np_points)
canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
```

### 5.4 主循环交互逻辑

```python
for e in window.get_events(ti.ui.PRESS):
    if e.key == ti.ui.LMB:          # 鼠标左键：添加控制点
        pos = window.get_cursor_pos()
        control_points.append(pos)
    elif e.key == 'c':              # C 键：清空所有控制点
        control_points = []
    elif e.key == 'b':              # B 键：切换贝塞尔 / B 样条模式
        mode = 'bspline' if mode == 'bezier' else 'bezier'
```

---

## 六、运行方式

```bash
uv run -m src.Work2.main
```

### 交互说明

| 操作       | 效果                  |
| -------- | ------------------- |
| 鼠标左键点击画布 | 添加控制点（红色圆点）         |
| 键盘 `C`   | 清空所有控制点，重置画布        |
| 键盘 `B`   | 在贝塞尔曲线模式与 B 样条模式间切换 |

---

## 七、效果演示

### 必做：贝塞尔曲线交互渲染（含反走样）

<img width="720" height="752" alt="必做1" src="https://github.com/user-attachments/assets/c33dfbe5-7a7e-4ca8-a747-0de2a4d71b23" />


逐步添加控制点后，程序实时绘制绿色贝塞尔曲线与灰色控制多边形。反走样通过 $3 \times 3$ 邻域距离加权实现，曲线边缘平滑无明显锯齿。按 `C` 键清空后可重新绘制。

### 选做：B 样条曲线模式切换

<img width="720" height="752" alt="选做" src="https://github.com/user-attachments/assets/b89e4606-cded-4e79-b5ce-cf144c4d33e2" />


按 `B` 键在两种模式间切换。在相同控制点下，贝塞尔曲线经过首尾控制点但受所有点影响（全局控制）；B 样条曲线不经过任何控制点，但移动局部控制点只影响曲线的对应一段（局部控制），体现了两种曲线方案的本质差异。

---

## 八、贝塞尔曲线与 B 样条曲线的对比

| 特性       | 贝塞尔曲线               | 均匀三次 B 样条          |
| -------- | ------------------- | ------------------ |
| 控制点影响范围  | 全局（移动任意一点影响整条曲线）    | 局部（每点只影响相邻 4 段）    |
| 是否经过控制点  | 经过首尾两点              | 不经过任何控制点           |
| 阶数与控制点关系 | $n$ 个点对应 $n-1$ 阶多项式 | 阶数固定（三次），控制点可任意增加  |
| 计算复杂度    | 随控制点增多急剧上升          | 分段线性，控制点增多时复杂度线性增长 |
| 适用场景     | 控制点较少的精确造型          | 控制点较多的复杂曲线设计       |

---

## 九、实验总结

1. **De Casteljau 算法**：递归线性插值是贝塞尔曲线的核心，其本质是伯恩斯坦多项式的几何构造形式，数值稳定性优于直接展开多项式求值。

2. **光栅化流程**：从连续浮点坐标到离散像素索引的映射是光栅化的核心操作，简单截断会导致走样，$3 \times 3$ 邻域距离加权可有效平滑边缘。

3. **CPU-GPU 协同**：批量传输（Batching）是 GPU 渲染的重要范式。将 1001 次 CPU 计算结果一次性通过 `from_numpy` 传入显存，再由 `@ti.kernel` 并行处理，相比逐点通信性能提升显著。

4. **原子操作**：反走样中多个曲线点可能同时贡献给同一像素，`ti.atomic_add` 保证了 GPU 并行写入的正确性，是并行渲染中处理写冲突的标准手段。

5. **B 样条的局部控制性**：通过实验直观验证了 B 样条的局部控制特性，在密集添加控制点时，曲线只在局部发生形变，而贝塞尔曲线会整体变形，这是 B 样条在 CAD 和字体设计领域被广泛采用的根本原因。
