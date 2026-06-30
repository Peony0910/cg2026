# 实验五：Whitted-Style 光线追踪（GIF图在第九部分，需要稍等片刻加载）

**202411081008-冯丹蕊-计算机科学与技术（公费师范）**

---

## 一、实验目标

1. 理解光线投射（Ray Casting）与光线追踪（Ray Tracing）的本质区别；
2. 掌握通过发射次级射线（Secondary Rays）实现硬阴影与理想镜面反射的方法；
3. 学习如何将传统递归光线追踪算法改写为适合 GPU 并行计算的迭代模式；
4. 选做：基于斯涅尔定律实现玻璃折射材质，以及基于多重采样的抗锯齿（MSAA）。

---

## 二、实验原理

### 2.1 光线投射 vs 光线追踪

| 特性 | 光线投射（Ray Casting） | 光线追踪（Ray Tracing） |
|------|------------------------|------------------------|
| 射线类型 | 仅主射线（Primary Ray） | 主射线 + 次级射线（Shadow / Reflect / Refract） |
| 光照模型 | 局部光照（Phong） | 全局光照（阴影、反射、折射） |
| 物理正确性 | 近似 | 更接近物理真实 |
| 计算代价 | 低 | 高（每次弹射生成新射线） |

本实验采用 Whitted-Style 光线追踪：主射线击中物体后根据材质类型分支，漫反射物体计算局部光照并终止，镜面物体生成反射射线继续传播，玻璃物体根据菲涅耳公式在反射与折射之间随机选择。

### 2.2 迭代式光线弹射

GPU 不支持递归（调用栈深度不确定），将递归改写为固定次数的 `for` 循环：

```

初始化：
throughput = (1.0, 1.0, 1.0)   # 光线能量衰减系数
final_color = (0.0, 0.0, 0.0)  # 累积颜色

for bounce in range(max_bounces):
场景求交
未命中 → 加背景色，break
命中镜面 → 更新射线，throughput × 0.8，continue
命中玻璃 → 折射/反射分支，throughput × 0.9，continue
命中漫反射 → 阴影测试 + 局部光照，final_color += throughput × 直接光照，break

```

每次弹射乘以衰减系数，保证能量守恒，防止无限反射导致过曝。

### 2.3 反射向量

$$\hat{R} = \hat{I} - 2(\hat{I} \cdot \hat{N})\hat{N}$$

其中 $\hat{I}$ 为入射方向，$\hat{N}$ 为表面法向量（单位向量）。

### 2.4 折射向量（斯涅尔定律，选做）

斯涅尔定律描述光线在两种介质界面处的折射关系：

$$\eta_1 \sin\theta_1 = \eta_2 \sin\theta_2$$

折射方向向量的计算（向量形式）：

$$\hat{T} = \frac{\eta_1}{\eta_2}\hat{I} + \left(\frac{\eta_1}{\eta_2}\cos\theta_i - \cos\theta_t\right)\hat{N}$$

其中 $\cos\theta_i = -\hat{N} \cdot \hat{I}$，$\cos\theta_t = \sqrt{1 - \left(\frac{\eta_1}{\eta_2}\right)^2(1 - \cos^2\theta_i)}$。

当根号内为负数时，发生**全内反射**，此时不产生折射射线，改为反射处理。

### 2.5 菲涅耳近似（Schlick 公式，选做）

真实玻璃的反射与折射比例随入射角变化，用 Schlick 近似：

$$F(\theta) = F_0 + (1 - F_0)(1 - \cos\theta)^5, \quad F_0 = \left(\frac{\eta - 1}{\eta + 1}\right)^2$$

以随机数与 $F(\theta)$ 比较决定射线走反射还是折射路径，在统计意义上逼近真实菲涅耳效果。

### 2.6 抗锯齿 MSAA（选做）

对每个像素内随机偏移发射 $n$ 条主射线，将颜色平均：

$$C_{pixel} = \frac{1}{n}\sum_{k=1}^{n} C(\hat{r}_k)$$

其中 $\hat{r}_k$ 的像素内偏移为 $[-0.5, 0.5]$ 均匀随机采样。采样数越多边缘越平滑，但帧率线性下降，实验默认 4 倍采样。

---

## 三、项目结构

```

CG-Lab/
├── src/
│   └── Work4/
│       ├── **init**.py
│       └── main.py        # 完整实现：光线追踪 + 玻璃折射 + MSAA
├── assets/
│   ├── raytrace_demo.gif  # 必做效果演示
│   └── glass_demo.gif     # 选做玻璃材质演示
└── README.md

````

---

## 四、场景设置

| 元素 | 参数 |
|------|------|
| 地板平面 | $y = -1.0$，法线 $(0,1,0)$，黑白棋盘格漫反射 |
| 红色漫反射球（可切换为玻璃） | 圆心 $(-1.2,\ 0.0,\ 0)$，半径 $1.0$ |
| 银色镜面球 | 圆心 $(1.2,\ 0.0,\ 0)$，半径 $1.0$，颜色 $(0.9,0.9,0.9)$ |
| 摄像机 | 位于 $(0,\ 1,\ 5)$，朝向 $-Z$ 方向，轻微俯角 |
| 点光源 | 初始位于 $(2,\ 4,\ 3)$，通过 UI 动态调节 |
| 背景颜色 | 深青色 $(0.05,\ 0.15,\ 0.2)$ |

---

## 五、核心代码说明

### 5.1 场景统一求交

将所有几何体的求交封装在一个函数中，返回最近交点的距离、法向量、颜色和材质 ID，实现深度测试：

```python
@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n   = ti.Vector([0.0, 0.0, 0.0])
    hit_c   = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 左侧球（漫反射/玻璃动态切换）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        if use_glass[None] == 1:
            hit_c = ti.Vector([1.0, 1.0, 1.0])
            hit_mat = MAT_GLASS
        else:
            hit_c = ti.Vector([0.8, 0.1, 0.1])
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
        ix = ti.cast(ti.floor(p.x * 2.0), ti.i32)
        iz = ti.cast(ti.floor(p.z * 2.0), ti.i32)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat
````

棋盘格纹理通过将交点 $x$、$z$ 坐标乘以网格缩放系数后取整，判断两个整数索引之和的奇偶性来分配颜色，无需任何纹理贴图。

### 5.2 折射计算（选做）

```python
@ti.func
def refract(I, N, eta):
    cos_theta = -N.dot(I)
    N_local = N
    eta_local = eta
    if cos_theta < 0.0:           # 射线从内部射出
        N_local = -N
        cos_theta = -N_local.dot(I)
        eta_local = 1.0 / eta     # 折射率取倒数
    sin_theta2 = eta_local * eta_local * (1.0 - cos_theta * cos_theta)
    result = ti.Vector([0.0, 0.0, 0.0])
    if sin_theta2 <= 1.0:         # 未发生全内反射
        cos_phi = ti.sqrt(1.0 - sin_theta2)
        R = eta_local * I + (eta_local * cos_theta - cos_phi) * N_local
        result = normalize(R)
    return result                 # 返回零向量表示全内反射
```

函数同时处理从外向内（折射率 $\eta$）和从内向外（折射率 $1/\eta$）两种情况，全内反射时返回零向量，调用方检测后切换为反射处理。

### 5.3 迭代弹射主循环

```python
for bounce in range(max_bounces[None]):
    t, N, obj_color, mat_id = scene_intersect(ro, rd)

    if t > 1e9:                        # 未命中，加背景色
        final_color += throughput * bg_color
        break

    p = ro + rd * t

    if mat_id == MAT_MIRROR:
        ro = p + N * 1e-4              # 偏移防止自相交
        rd = reflect(rd, N)
        throughput *= 0.8 * obj_color  # 镜面能量衰减
        continue

    elif mat_id == MAT_GLASS:
        refl_dir = reflect(rd, N)
        refr_dir = refract(rd, N, 1.5)
        if refr_dir.norm() < 1e-6:     # 全内反射
            ro = p + N * 1e-4
            rd = refl_dir
        else:
            cos_theta = ti.abs(N.dot(rd))
            fresnel = 0.04 + 0.96 * ti.pow(1.0 - cos_theta, 5.0)
            if ti.random(ti.f32) < fresnel:
                ro = p + N * 1e-4      # 反射
                rd = refl_dir
            else:
                ro = p - N * 1e-4      # 折射（向内偏移）
                rd = refr_dir
        throughput *= 0.9 * obj_color
        continue

    else:                              # MAT_DIFFUSE
        L = normalize(light_pos - p)
        shadow_orig = p + N * 1e-4
        dist_to_light = (light_pos - p).norm()
        in_shadow = shadow_ray(shadow_orig, L, dist_to_light)

        ambient = 0.2 * obj_color
        direct_light = ambient
        if not in_shadow:
            diff = ti.max(0.0, N.dot(L))
            direct_light += 0.8 * diff * obj_color

        final_color += throughput * direct_light
        break
```

### 5.4 MSAA 多重采样（选做）

```python
for _ in range(n_samples):
    dx = ti.random(ti.f32) - 0.5      # 像素内随机抖动
    dy = ti.random(ti.f32) - 0.5
    u = (i + dx - res_x / 2.0) / res_y * 2.0
    v = (j + dy - res_y / 2.0) / res_y * 2.0
    # ... 完整追踪流程 ...
    color_sum += final_color

pixels[i, j] = ti.math.clamp(color_sum / n_samples, 0.0, 1.0)
```

---

## 六、Shadow Acne 问题与解决

Shadow Acne（自相交噪点）是光线追踪实现中最常见的数值精度问题。

**成因：** 交点 $P$ 在浮点精度下并不精确落在几何体表面，而是略微偏内或偏外。若直接从 $P$ 出发发射暗影射线或反射射线，射线极有可能立即与自身表面再次相交（$t \approx 0$），判定为被遮挡，产生随机黑色噪点。

**解决方案：** 将次级射线起点沿法向量方向偏移一个极小量 $\varepsilon$：

$$P_{shadow} = P + \hat{N} \cdot \varepsilon, \quad \varepsilon = 10^{-4}$$

折射射线穿入介质内部时需反向偏移：

$$P_{refract} = P - \hat{N} \cdot \varepsilon$$

---

## 七、UI 交互面板

```python
with gui.sub_window("Controls", 0.70, 0.05, 0.28, 0.32):
    light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
    light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None],  1.0, 8.0)
    light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
    max_bounces[None] = gui.slider_int('Max Bounces',       max_bounces[None],      1,  5)
    samples_per_pixel[None] = gui.slider_int('MSAA Samples', samples_per_pixel[None], 1, 16)
    if gui.button("Switch to Glass" if use_glass[None] == 0 else "Switch to Diffuse"):
        use_glass[None] = 1 - use_glass[None]
```

| 控件           | 范围                     | 默认值       | 可观察效果                    |
| ------------ | ---------------------- | --------- | ------------------------ |
| Light X/Y/Z  | X: -5~5，Y: 1~8，Z: -5~5 | (2, 4, 3) | 阴影随光源位置实时移动              |
| Max Bounces  | 1 ~ 5                  | 3         | 设为 1 时镜面球显示纯黑，≥2 时出现反射场景 |
| MSAA Samples | 1 ~ 16                 | 4         | 增大采样数边缘锯齿减少，帧率线性下降       |
| 切换按钮         | —                      | 漫反射       | 切换左球为玻璃材质，观察折射与菲涅耳效果     |

---
<img width="803" height="639" alt="cbc37a225d49590195cf2f7cebb42340" src="https://github.com/user-attachments/assets/137cea6b-730f-49b8-b28b-3926268d99f1" />

## 八、运行方式

```bash
uv run -m src.Work4.main
```

---

## 九、效果演示

### 必做：硬阴影 + 镜面反射
<img width="730" height="576" alt="必做1" src="https://github.com/user-attachments/assets/ab57b637-f569-4e0e-9e9f-b78e669d97e3" />
<img width="730" height="576" alt="必做2_edited" src="https://github.com/user-attachments/assets/d196d07f-d38d-4939-b8c5-df3ccfcff7f7" />
<img width="912" height="720" alt="必做3_edited" src="https://github.com/user-attachments/assets/d1f9a466-8c7b-4cd8-8150-4b6b7e2a7155" />


右侧镜面球反射出场景中的红球、棋盘格地板与背景；拖动 Light X/Y/Z 滑动条，阴影实时跟随光源移动；将 Max Bounces 调为 1 时镜面球变黑（无弹射），调大后可见镜中镜的多次反射效果。

### 选做：玻璃折射 + MSAA 抗锯齿

<img width="717" height="576" alt="选做1_edited (1)" src="https://github.com/user-attachments/assets/dc1968d4-e5a3-40a1-b2a3-960085068dc6" />
<img width="730" height="576" alt="选做2_edited" src="https://github.com/user-attachments/assets/fb92a56b-ea9e-43a1-b526-fc2a5e7d3f77" />


切换至玻璃模式后，左球变为透明玻璃材质：球体边缘出现菲涅耳反射高光，球体内部可见折射后的背景场景，大入射角（掠射）处反射比例明显升高。将 MSAA Samples 从 1 调至 8，物体边缘的阶梯状锯齿逐渐平滑为连续过渡。

---

## 十、Phong 光照与光线追踪的对比总结

| 维度     | Phong 光照（实验三） | Whitted 光线追踪（本实验） |
| ------ | ------------- | ----------------- |
| 阴影     | 通过暗影射线实现      | 通过暗影射线实现（相同）      |
| 反射     | 无             | 通过反射射线迭代弹射        |
| 折射     | 无             | 斯涅尔定律 + 菲涅耳近似     |
| 全局光照   | 不支持（仅局部）      | 部分支持（镜面全局，漫反射仍局部） |
| GPU 适配 | 天然并行，无分支问题    | 需将递归改写为迭代，分支较复杂   |

---

## 十一、实验总结

1. **递归改迭代**：GPU 不支持不定深度递归，用固定次数 `for` 循环 + `throughput` 累乘替代，是 GPU 光线追踪编程的核心范式，吞吐量系数的物理意义是光传播路径上的累计能量衰减。

2. **Shadow Acne 的本质**：浮点精度导致交点偏移，沿法向量偏移 $10^{-4}$ 是光线追踪实现中处理自相交的标准工程手段，偏移量过小无效，过大会导致阴影悬浮。

3. **菲涅耳效应的直观感受**：玻璃球在正视方向几乎全透明，在掠射角处出现明显反射，这与日常观察窗玻璃的体验一致，Schlick 近似在保持物理趋势的同时将计算代价压到最低。

4. **MSAA 的本质**：多重采样本质上是对像素内的颜色函数做蒙特卡洛积分，随机抖动保证采样点分布均匀，采样数越多方差越小，边缘越平滑，但帧率代价与采样数严格线性相关。

5. **材质系统的设计价值**：用整数材质 ID 区分漫反射、镜面、玻璃，在一个求交函数中统一返回材质信息，使主循环的分支逻辑清晰可扩展，是实际渲染引擎材质系统的简化原型。
