# 实验三：Phong 光照模型与光线投射渲染（GIF图在第八部分，需要稍等片刻加载）

**202411081008-冯丹蕊-计算机科学与技术（公费师范）**
---

## 一、实验目标

1. 理解并掌握局部光照的基本原理，区分环境光（Ambient）、漫反射（Diffuse）和镜面高光（Specular）三个分量；
2. 熟练掌握三维空间中的向量运算：法向量计算、光线方向、视线方向与反射向量；
3. 掌握光线投射（Ray Casting）与隐式几何体求交的实现方法；
4. 掌握如何利用 Taichi 实现交互式渲染，通过 UI 控件实时调节材质参数；
5. 选做：实现 Blinn-Phong 模型升级与硬阴影（Shadow Ray）效果。

---

## 二、实验原理

### 2.1 Phong 光照模型

Phong 模型将物体表面反射的光分为三个独立分量叠加：

$$I = I_{ambient} + I_{diffuse} + I_{specular}$$

**环境光（Ambient）**

模拟场景中经多次反射后均匀分布的背景光，与观察方向和光照方向无关：

$$I_{ambient} = K_a \cdot I_{light} \cdot C_{object}$$

**漫反射（Diffuse）**

模拟粗糙表面向各方向均匀散射的光，遵循 Lambert 余弦定律，强度与入射角余弦值成正比：

$$I_{diffuse} = K_d \cdot \max(0,\ \hat{N} \cdot \hat{L}) \cdot I_{light} \cdot C_{object}$$

**镜面高光（Specular）**

模拟光滑表面的强光反射，强度与视线方向 $\hat{V}$ 和理想反射方向 $\hat{R}$ 的夹角相关：

$$I_{specular} = K_s \cdot \max(0,\ \hat{R} \cdot \hat{V})^{n} \cdot I_{light}$$

其中反射向量 $\hat{R} = \hat{I} - 2(\hat{I} \cdot \hat{N})\hat{N}$，$n$ 为高光指数（Shininess）。

各符号含义如下：

| 符号 | 含义 |
|------|------|
| $\hat{N}$ | 表面法向量（单位向量） |
| $\hat{L}$ | 指向光源的方向向量（单位向量） |
| $\hat{V}$ | 指向摄像机的方向向量（单位向量） |
| $\hat{R}$ | 光线的理想反射向量（单位向量） |
| $n$ | 高光指数 Shininess |
| $K_a, K_d, K_s$ | 环境光、漫反射、镜面高光系数 |

### 2.2 Blinn-Phong 模型（选做）

Blinn-Phong 模型用半程向量 $\hat{H}$ 替代反射向量 $\hat{R}$，计算更高效且在大入射角时表现更自然：

$$\hat{H} = \frac{\hat{L} + \hat{V}}{|\hat{L} + \hat{V}|}$$

$$I_{specular}^{Blinn} = K_s \cdot \max(0,\ \hat{N} \cdot \hat{H})^{n} \cdot I_{light}$$

**Phong 与 Blinn-Phong 的视觉差异：**

- 在观察角与反射角接近时（小入射角），两者结果相近；
- 在大入射角（掠射角）时，Phong 模型的高光区域会出现突然截断（$\hat{R} \cdot \hat{V}$ 变为负数被 `max` 截零），高光边缘出现硬边；
- Blinn-Phong 在同等情况下过渡更平滑，高光形状更自然，这也是实时渲染管线普遍采用 Blinn-Phong 的原因。

### 2.3 硬阴影（Hard Shadow，选做）

在交点 $P$ 处向光源方向发射一条暗影射线（Shadow Ray）。若该射线在到达光源之前击中其他几何体，则 $P$ 处于阴影中，此时只计算环境光：

$$I_{shadow} = I_{ambient}$$

---

## 三、项目结构

```

CG-Lab/
├── src/
│   └── Work3/
│       ├── **init**.py
│       └── main.py       # 完整实现：Phong/Blinn-Phong + 硬阴影 + UI 控件
├── assets/
│   └── phong_demo.gif    # 效果演示
└── README.md

````

---

## 四、场景设置

| 元素 | 参数 |
|------|------|
| 红色球体 | 圆心 $(-1.2,\ -0.2,\ 0)$，半径 $1.2$，颜色 $(0.8,\ 0.1,\ 0.1)$ |
| 紫色圆锥 | 顶点 $(1.2,\ 1.2,\ 0)$，底面 $y=-1.4$，底面半径 $1.2$，颜色 $(0.6,\ 0.2,\ 0.8)$ |
| 摄像机 | 固定于 $(0,\ 0,\ 5)$，朝向 $-Z$ 方向 |
| 点光源 | 位于 $(2,\ 3,\ 4)$，颜色纯白 $(1.0,\ 1.0,\ 1.0)$ |
| 背景颜色 | 深青色 $(0.05,\ 0.15,\ 0.15)$ |

---

## 五、核心算法说明

### 5.1 光线投射与像素坐标映射

为屏幕上每个像素 $(i, j)$ 生成一条射线，起点为摄像机位置 $ro$，方向由像素归一化坐标决定：

```python
u = (i - res_x / 2.0) / res_y * 2.0
v = (j - res_y / 2.0) / res_y * 2.0
ro = ti.Vector([0.0, 0.0, 5.0])
rd = normalize(ti.Vector([u, v, -1.0]))
````

除以 `res_y`（而非各自的宽高）保证了水平与垂直方向的像素比例一致，避免图像拉伸变形。

### 5.2 球体求交

射线 $P(t) = ro + t \cdot rd$ 与球心 $C$、半径 $r$ 的球的交点满足：

$$|P(t) - C|^2 = r^2$$

展开得关于 $t$ 的一元二次方程：

$$t^2 + 2(oc \cdot rd),t + (oc \cdot oc - r^2) = 0, \quad oc = ro - C$$

取最小正根 $t_1$ 为最近交点，法向量为：

$$\hat{N} = \frac{P(t) - C}{|P(t) - C|}$$

```python
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
```

### 5.3 圆锥求交

圆锥顶点为 $apex$，向下扩张，底面在 $y = base_y$，半高 $H = apex.y - base_y$，底面半径 $r$，锥面满足：

$$x^2 + z^2 = k \cdot (y - apex.y)^2, \quad k = \left(\frac{r}{H}\right)^2$$

将射线代入得二次方程，取满足 $-H \le y_{local} \le 0$ 范围内的最小正根为有效交点，圆锥侧面法向量为：

$$\hat{N} = \text{normalize}(x_{local},\ -k \cdot y_{local},\ z_{local})$$

```python
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
```

### 5.4 深度测试（Z-buffer）

对每个像素同时计算射线与球体、圆锥的交点距离，取最小正 $t$ 对应的物体进行着色，保证正确的遮挡关系：

```python
min_t = 1e10
# 球体
t_sph, n_sph = intersect_sphere(...)
if 0 < t_sph < min_t:
    min_t = t_sph
    hit_normal = n_sph
    hit_color = ti.Vector([0.8, 0.1, 0.1])
    hit_pos = ro + rd * min_t
# 圆锥
t_cone, n_cone = intersect_cone(...)
if 0 < t_cone < min_t:
    min_t = t_cone
    hit_normal = n_cone
    hit_color = ti.Vector([0.6, 0.2, 0.8])
    hit_pos = ro + rd * min_t
```

### 5.5 Phong 着色计算

在最近交点处计算三个光照分量：

```python
N = hit_normal                      # 法向量（已归一化）
L = normalize(light_pos - P)        # 指向光源
V = normalize(ro - P)               # 指向摄像机

# 环境光
ambient = Ka[None] * light_color * hit_color

# 漫反射（截断负值）
diff = ti.max(0.0, N.dot(L))
diffuse = Kd[None] * diff * light_color * hit_color

# 镜面高光
# Phong 模式
R = normalize(reflect(-L, N))
spec_val = ti.max(0.0, R.dot(V)) ** shininess[None]
# Blinn-Phong 模式
H = normalize(L + V)
spec_val = ti.max(0.0, N.dot(H)) ** shininess[None]

specular = Ks[None] * spec_val * light_color

# 最终颜色（防止过曝）
color = ti.math.clamp(ambient + diffuse + specular, 0.0, 1.0)
```

### 5.6 硬阴影（选做）

从交点 $P$ 沿光源方向发射暗影射线，偏移 $1 \times 10^{-4}$ 避免自相交（数值精度问题）：

```python
@ti.func
def shadow_ray(ro, rd, t_max):
    hit = False
    t_sph, _ = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
    if 0 < t_sph < t_max:
        hit = True
    if not hit:
        t_cone, _ = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < t_max:
            hit = True
    return hit

# 主渲染内核中
shadow_origin = P + N * 1e-4
shadow_t_max  = (light_pos - shadow_origin).norm()
if shadow_ray(shadow_origin, L, shadow_t_max):
    in_shadow = True
# 阴影区域只保留环境光
if in_shadow:
    color = ambient
```

---

## 六、UI 交互面板

使用 `ti.ui.Window` 的 `gui.sub_window` 创建浮动参数面板，四个滑动条实时绑定着色器参数：

| 参数         | 范围          | 默认值  | 效果                |
| ---------- | ----------- | ---- | ----------------- |
| Ka（环境光系数）  | 0.0 ~ 1.0   | 0.2  | 控制暗部基础亮度，过高会损失立体感 |
| Kd（漫反射系数）  | 0.0 ~ 1.0   | 0.7  | 控制表面主体明暗对比        |
| Ks（镜面高光系数） | 0.0 ~ 1.0   | 0.5  | 控制高光亮斑强度          |
| N（高光指数）    | 1.0 ~ 128.0 | 32.0 | 越大高光越集中，越小高光越散    |

此外提供一个切换按钮，在 Phong 与 Blinn-Phong 模式间实时切换：

```python
with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.25):
    Ka[None]        = gui.slider_float('Ka (Ambient)',   Ka[None],        0.0,   1.0)
    Kd[None]        = gui.slider_float('Kd (Diffuse)',   Kd[None],        0.0,   1.0)
    Ks[None]        = gui.slider_float('Ks (Specular)',  Ks[None],        0.0,   1.0)
    shininess[None] = gui.slider_float('N (Shininess)',  shininess[None], 1.0, 128.0)
    if gui.button("Switch to Blinn-Phong" if use_blinn[None] == 0
                  else "Switch to Phong"):
        use_blinn[None] = 1 - use_blinn[None]
```

---

## 七、运行方式

```bash
uv run -m src.Work3.main
```

程序启动后弹出 800×600 渲染窗口，右上角提供参数控制面板，调节任意滑动条后画面实时更新。
<img width="803" height="636" alt="e23b5a84b93690a4a782a593f305f529" src="https://github.com/user-attachments/assets/a6712c15-58c9-4c9d-8250-e610880fc087" />

---

## 八、效果演示
必做效果（4张gif图）：
1. Ka (环境光系数): 范围 0.0 ~ 1.0，默认值 0.2。
<img width="912" height="720" alt="1" src="https://github.com/user-attachments/assets/3bfb412b-0bea-4af5-a3ec-9306fba8e13f" />
2. Kd (漫反射系数): 范围 0.0 ~ 1.0，默认值 0.7。
<img width="1008" height="720" alt="2" src="https://github.com/user-attachments/assets/46c6670a-c307-4159-a08a-3e2218278ea9" />
3. Ks (镜面高光系数): 范围 0.0 ~ 1.0，默认值 0.5。
<img width="960" height="720" alt="3" src="https://github.com/user-attachments/assets/6d9b56a4-729f-4d94-80ad-b9beeb60085a" />
4. Shininess (高光指数): 范围 1.0 ~ 128.0，默认值 32.0。
<img width="976" height="720" alt="4" src="https://github.com/user-attachments/assets/71a58d8c-605f-40bd-841d-86bd55d0f4fe" />

选做效果：
<img width="928" height="720" alt="选做1" src="https://github.com/user-attachments/assets/02e7e8c3-82c9-4907-b1e6-00738a3c3c34" />


演示内容说明：

* 左侧红色球体和右侧紫色圆锥均受同一点光源照射，深度测试保证两者在重叠区域的正确遮挡；
* 调节 Ka 可观察暗部亮度变化；调节 Kd 改变漫反射强度；调节 Ks 和 Shininess 可直观感受高光集中程度的变化；
* 切换到 Blinn-Phong 模式后，高光边缘在大入射角处过渡更平滑，不再出现 Phong 模型的硬截断；
* 硬阴影效果下，圆锥在球体上投射出清晰的阴影边界，阴影内仅保留环境光照亮。

---

## 九、常见问题与解决方案

| 现象         | 原因                                                     | 解决方法                                     |
| ---------- | ------------------------------------------------------ | ---------------------------------------- |
| 渲染结果全黑     | 参与点乘的向量未归一化                                            | 所有方向向量使用 `normalize()` 处理                |
| 出现黑色噪点或马赛克 | $\hat{N} \cdot \hat{L}$ 或 $\hat{R} \cdot \hat{V}$ 出现负值 | 使用 `ti.max(0.0, dot)` 截断负值               |
| 颜色过曝发白     | RGB 各分量叠加超过 1.0                                        | 写入像素前使用 `ti.math.clamp(color, 0.0, 1.0)` |
| 阴影处出现自相交噪点 | 暗影射线起点与交点重合导致自遮挡                                       | 起点沿法向量偏移 `P + N * 1e-4`                  |

---

## 十、实验总结

1. **Phong 模型的分量理解**：三个分量各有物理对应，环境光保底、漫反射决定立体感、高光体现材质光泽度。只有三者合理配比才能产生真实感光照效果，仅靠任意单一分量均无法达到满意结果。

2. **向量归一化的重要性**：光照计算中 $\hat{N}$、$\hat{L}$、$\hat{V}$、$\hat{H}$ 必须是单位向量，否则点乘结果不等于余弦值，整个光照公式的物理意义失效。归一化是着色器编程中最容易遗漏的细节。

3. **深度测试的本质**：光线与多个物体的交点距离 $t$ 直接对应相机空间的深度，取最小正 $t$ 等价于 Z-buffer 深度测试，是保证正确遮挡关系的核心机制。

4. **Phong 与 Blinn-Phong 的差异**：Blinn-Phong 用半程向量替代反射向量，在大入射角时避免了 Phong 模型的高光硬截断，视觉过渡更自然，且计算反射向量需要两次点乘而半程向量只需一次归一化，实际性能也更优。

5. **硬阴影的数值稳定性**：暗影射线的起点必须沿法向量微小偏移，否则交点自身会被当作遮挡物，产生随机噪点。这是光线追踪中处理自相交（Self-intersection）的标准做法。
