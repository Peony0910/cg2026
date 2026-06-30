# 实验六：可微渲染与网格优化

**202411081008-冯丹蕊-计算机科学与技术（公费师范）**

---

## 一、实验目标

1. 理解可微光栅化的原理，掌握软光栅化（Soft Rasterization）在处理离散几何体边界时的数学近似方法；
2. 掌握通过多视角二维图像（剪影 / RGB）反推并优化三维网格顶点坐标的完整流程；
3. 深刻理解正则化对于防止网格拓扑崩坏和陷入局部最优的决定性作用；
4. 选做：基于 SoftPhongShader 实现联合纹理优化，同时拟合剪影与 RGB 图像，优化顶点坐标与顶点颜色。

---

## 二、实验原理

### 2.1 软光栅化（Soft Rasterization）

传统硬光栅化中，像素的覆盖判断是非 0 即 1 的阶跃函数，导致边界处梯度为零（梯度消失），优化器无法获得有效梯度信号。

软光栅化通过计算像素到三角形边缘的有符号距离 $d$，使用 Sigmoid 函数产生平滑的概率过渡：

$$A(d) = \text{sigmoid}\left(\frac{d}{\sigma}\right) = \frac{1}{1 + e^{-d/\sigma}}$$

其中 $\sigma$ 控制边缘模糊程度。即使顶点在像素外部，也能提供非零梯度，引导顶点向正确方向移动。$\sigma$ 越小越接近硬光栅化，$\sigma$ 越大梯度越平滑但图像越模糊。

### 2.2 网格正则化（Mesh Regularization）

仅依靠图像 Loss 驱动顶点移动，会导致顶点相互交叉形成"刺猬"状拓扑，陷入局部最优。必须引入三种正则化约束：

**拉普拉斯平滑（Laplacian Smoothing）**

约束每个顶点与其相邻顶点的坐标均值之差尽量小，防止表面出现尖锐突起：

$$L_{lap} = \sum_{i} \left\| v_i - \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} v_j \right\|^2$$

**边长一致性（Edge Length Penalty）**

惩罚过长或过短的边，防止三角形严重拉伸变形：

$$L_{edge} = \sum_{(i,j) \in \text{edges}} \|v_i - v_j\|^2$$

**法线一致性（Normal Consistency）**

约束相邻三角形面的法线方向接近，保持表面全局平滑：

$$L_{normal} = \sum_{\text{adjacent faces}} (1 - \hat{n}_i \cdot \hat{n}_j)$$

**总 Loss：**

$$L_{total} = L_{silhouette} + w_{lap}L_{lap} + w_{edge}L_{edge} + w_{normal}L_{normal}$$

### 2.3 联合纹理优化原理（选做）

在形状优化基础上，引入 SoftPhongShader 渲染 RGB 图像，将顶点颜色 $C \in [0,1]^3$ 设为可微参数，同时优化形状和颜色：

$$L_{total} = w_{sil} \cdot L_{silhouette} + w_{rgb} \cdot L_{rgb} + w_{lap}L_{lap} + w_{edge}L_{edge} + w_{normal}L_{normal}$$

顶点颜色参数 `deform_colors` 在 logit 空间优化，通过 $\text{sigmoid}$ 保证值域合法：

$$C = \text{sigmoid}(\theta_c), \quad \theta_c \in \mathbb{R}^{V \times 3}$$

---

## 三、项目结构

```

src/Work5/
├── README.md                          # 实验报告（含选做说明）
├── yunxing.ipynb                      # 必做+选做代码
├── output_meshes/                     # 必做结果文件夹
│   ├── mesh_epoch_000.obj
│   ├── mesh_epoch_020.obj
│   └── ... (所有 .obj 文件)
├── output_meshes_bonus/               # 选做结果文件夹
│   └── ... (选做生成的 .obj 文件)
├── bonus_final_result.png             # 选做最终结果截图
├── bonus_loss_curve.png               # 选做损失曲线截图
├── work5_result.png                   # 必做最终结果截图
└── work5_training.png                 # 必做训练过程截图

````

---

## 四、必做：基于剪影的网格优化

### 4.1 场景与渲染配置

| 参数 | 值 |
|------|-----|
| 目标模型 | cow.obj（已中心归一化） |
| 摄像机视角数 | 20（方位角均匀分布 -180° ~ 180°） |
| 摄像机距离 | 2.7 |
| 渲染分辨率 | 256 × 256 |
| 软光栅化 $\sigma$ | $10^{-4}$ |
| `faces_per_pixel` | 50 |
| 初始源网格 | `ico_sphere(4)`（2562 顶点） |

### 4.2 损失函数与权重设置

```python
loss = loss_silhouette          \   # 剪影 MSE
     + 1.0 * loss_laplacian     \   # 拉普拉斯平滑
     + 0.1 * loss_edge          \   # 边长一致性
     + 0.01 * loss_normal           # 法线一致性
````

### 4.3 优化器配置

```python
optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
epochs = 300
```

### 4.4 核心代码说明

**目标剪影生成：**

```python
verts, faces, _ = load_obj("cow.obj")
verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])  # 归一化到单位球内
cow_mesh = Meshes(verts=[verts], faces=[faces_idx])

target_silhouette = shader(
    rasterizer(cow_mesh.extend(num_views)),
    cow_mesh.extend(num_views)
)[..., 3]  # 取 alpha 通道
```

**优化循环（核心）：**

```python
for i in range(epochs):
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    pred_silhouette = shader(
        rasterizer(new_src_mesh.extend(num_views)),
        new_src_mesh.extend(num_views)
    )[..., 3]

    loss_silhouette = ((pred_silhouette - target_silhouette) ** 2).mean()
    loss = loss_silhouette \
         + 1.0 * mesh_laplacian_smoothing(new_src_mesh) \
         + 0.1 * mesh_edge_loss(new_src_mesh) \
         + 0.01 * mesh_normal_consistency(new_src_mesh)

    loss.backward()
    optimizer.step()
```

---

## 五、选做：联合纹理优化（形状 + 顶点颜色）

### 5.1 初版实现的问题与调试过程

#### 5.1.1 问题一：正则化权重过大，导致形状完全无法收敛

初版代码沿用必做的正则化权重（$w_{lap}=1.0$，$w_{edge}=0.1$，$w_{normal}=0.01$），同时从第 0 轮就一并优化形状和颜色。运行后出现以下现象：

* 渲染结果完全不像奶牛，保持球形甚至更差；
* Loss 曲线中 Normal Consistency 从第 0 轮就飙升至 1.0 以上，主导了整个 Total Loss；
* 优化器的梯度被正则化项压制，剪影 Loss 几乎没有下降。
<img width="1305" height="477" alt="e83f6ad07fd7b24c1d7cde394073db6e" src="https://github.com/user-attachments/assets/a471be70-7a18-4fd4-835b-6ed2c05c74b7" />

**根本原因分析：** 从球体出发形变时，顶点大幅移动会瞬间产生很高的法线不一致惩罚。若 $w_{normal}$ 过大，优化器宁可保持球形不动也不愿承受法线惩罚，梯度信号被完全抑制。

下图为初版运行的 Loss 曲线，可以看到 Normal Consistency（绿线）从一开始就居高不下，Total Loss 几乎完全被其主导：


对应的渲染结果形状几乎没有优化，颜色更是一片混乱：
<img width="1330" height="627" alt="b41c285d2d7f91280cbbb077f8e37ef7" src="https://github.com/user-attachments/assets/a2525a7a-c9a9-4f51-a731-9b2ae546b95c" />


#### 5.1.2 问题二：形状与颜色同步优化互相干扰

在形状尚未收敛（仍是球体）时就引入 RGB Loss，颜色梯度与形状梯度相互竞争，导致两者都无法有效收敛。

#### 5.1.3 解决方案：两阶段优化 + 大幅降低正则化权重

针对上述两个问题，做出以下关键修改：

| 修改项          | 初版       | 修复版              |
| ------------ | -------- | ---------------- |
| $w_{normal}$ | 0.01     | 0.0001（降低 100 倍） |
| $w_{lap}$    | 1.0      | 0.1              |
| $w_{edge}$   | 0.1      | 0.01             |
| 优化策略         | 形状+颜色同步  | 两阶段：先形状后颜色       |
| 优化器          | Adam（单组） | Adam（形状/颜色独立学习率） |

### 5.2 两阶段优化策略

**阶段一（Epoch 0~299）：只优化形状**

仅将 `deform_verts` 传入优化器，颜色参数不参与梯度更新。权重设置为：

```python
w_sil    = 1.0
w_lap    = 0.1
w_edge   = 0.01
w_normal = 0.0001   # 关键：大幅降低，让剪影 Loss 主导

optimizer_shape = torch.optim.Adam([deform_verts_bonus], lr=1e-2)
```

**阶段二（Epoch 300~499）：联合优化形状 + 颜色**

形状已基本收敛后，引入 `deform_colors` 和 RGB Loss，使用独立学习率精细调节：

```python
w_sil    = 1.0
w_rgb    = 0.5    # 颜色 Loss 权重不宜过大，避免干扰形状
w_lap    = 0.1
w_edge   = 0.01
w_normal = 0.0001

optimizer_joint = torch.optim.Adam([
    {"params": deform_verts_bonus, "lr": 3e-3},  # 形状学习率降低，稳定微调
    {"params": deform_colors,      "lr": 1e-2},  # 颜色学习率稍高
])
```

### 5.3 顶点颜色参数化

顶点颜色在 logit 空间优化，保证值域合法：

```python
# 初始化为 0，sigmoid(0) = 0.5，即初始中性灰
deform_colors = torch.zeros((num_verts, 3), device=device, requires_grad=True)

# 优化过程中通过 sigmoid 映射到 (0,1)
current_colors = torch.sigmoid(deform_colors).unsqueeze(0)  # (1, V, 3)

new_mesh_colored = Meshes(
    verts=new_mesh_geo.verts_list(),
    faces=new_mesh_geo.faces_list(),
    textures=TexturesVertex(verts_features=current_colors),
)
```

### 5.4 目标 RGB 生成

用顶点坐标归一化作为奶牛的参考颜色，无需外部纹理贴图：

```python
cow_verts = cow_mesh.verts_packed()
cow_colors = (cow_verts - cow_verts.min(0)[0]) \
           / (cow_verts.max(0)[0] - cow_verts.min(0)[0] + 1e-8)

cow_mesh_colored = Meshes(
    verts=[cow_mesh.verts_list()[0]],
    faces=[cow_mesh.faces_list()[0]],
    textures=TexturesVertex(verts_features=cow_colors.unsqueeze(0)),
)

with torch.no_grad():
    target_rgb = rgb_renderer(cow_mesh_colored.extend(num_views))[..., :3]
```

---

## 六、实验结果

### 6.1 必做结果
初始：<img width="1058" height="567" alt="ae0af047c158236a814fec6188f0b0c7" src="https://github.com/user-attachments/assets/333298d1-6a0e-4269-be6f-578b70f51860" />

优化 300 轮后，球体逐渐变形为与奶牛轮廓高度吻合的网格，多视角剪影均与目标剪影接近。
<img width="1010" height="568" alt="bf7cac41cccda0b5b1ff3053b1ebaf6f" src="https://github.com/user-attachments/assets/fd29c1a9-03c9-41ba-9789-943bd3bdf001" />



### 6.2 选做 Loss 曲线分析

<img width="1292" height="479" alt="ab8c896c3076d0bf372e223ed47692f9" src="https://github.com/user-attachments/assets/d7fa2afd-378c-4536-911d-55ab0136be1d" />


**阶段一（0~300 轮）：**

* Total Loss 与 Silhouette Loss 几乎重合，从 0.22 迅速下降至接近 0，说明正则化权重设置合理，剪影 Loss 真正主导了优化过程；
* Laplacian 和 Edge Loss 平稳收敛，Normal Consistency 出现小幅震荡后稳定下降，网格拓扑保持健康。

**阶段二（300~500 轮）：**

* RGB Loss 引入后从 0 开始上升再逐渐下降，说明颜色参数正在学习目标分布；
* Normal Consistency 在阶段二切换时出现上升，原因是阶段二继续微调顶点位置引发局部法线变化，但仍在可控范围内；
* Total Loss 在阶段二小幅上升后趋于平稳，符合引入新 Loss 项后的预期行为。

### 6.3 选做最终渲染结果

<img width="1273" height="723" alt="e20fb9bd33d9e70e30a903a71d602b31" src="https://github.com/user-attachments/assets/5d293bcf-d81f-4b59-a9cc-59be0a9f0204" />


结果图按行排列：

* **第一行（GT Silhouette）：** 目标奶牛剪影，五个视角均呈现清晰的奶牛轮廓；
* **第二行（GT RGB）：** 目标奶牛 RGB 图，颜色由顶点坐标归一化得到，呈空间渐变分布；
* **第三行（Optimized RGB）：** 优化后网格的 RGB 渲染，形状轮廓与目标高度吻合，顶点颜色成功学出与目标接近的空间渐变分布。

---

## 七、参数分析与消融

| 参数                | 作用        | 过大的后果        | 过小的后果      |
| ----------------- | --------- | ------------ | ---------- |
| $w_{normal}$      | 约束相邻面法线一致 | 梯度被压制，形状无法形变 | 网格表面出现尖刺噪点 |
| $w_{lap}$         | 约束相邻顶点平滑  | 过于光滑，失去细节    | 顶点交叉，拓扑崩坏  |
| $w_{rgb}$         | 拟合目标颜色    | 干扰形状收敛       | 颜色学习缓慢     |
| $\sigma$（软光栅化）    | 边缘梯度平滑程度  | 图像过度模糊       | 梯度消失，优化失效  |
| `faces_per_pixel` | 软混合精度     | 显存占用增加       | 软光栅化效果退化   |

---

## 八、常见问题与解决

| 现象                   | 原因                                 | 解决方法                                        |
| -------------------- | ---------------------------------- | ------------------------------------------- |
| 形状完全不变，Loss 不下降      | 正则化权重过大，梯度被压制                      | 将 $w_{normal}$ 降至 $10^{-4}$ 量级              |
| 渲染结果是"刺猬"状           | 正则化权重过小，顶点交叉                       | 适当增大 $w_{lap}$                              |
| 颜色学不出来，一直发灰          | 颜色学习率过低或 $w_{rgb}$ 过小              | 颜色 `lr` 调至 $10^{-2}$，或增大 $w_{rgb}$          |
| 显存溢出（CUDA OOM）       | `num_views` 和 `faces_per_pixel` 过大 | 将 `num_views` 改为 10，`faces_per_pixel` 改为 20 |
| 阶段二 Normal Loss 突然升高 | 形状继续微调引发法线变化                       | 属于正常现象，降低阶段二顶点学习率至 $3\times10^{-3}$         |

---

## 九、实验总结

1. **软光栅化的核心价值**：用 Sigmoid 平滑边界的阶跃变化，将"是否在三角形内"这一离散判断转化为连续可微的概率值，这是可微渲染与传统渲染的本质区别。没有这一步，梯度无法通过像素传播到顶点坐标，整个优化流程就无从建立。

2. **正则化权重的决定性作用**：初版实验中，$w_{normal}=0.01$ 直接导致优化完全失败——Normal Consistency Loss 从第一轮就主导梯度，形状锁死在球体上。将其降至 $0.0001$ 后，剪影 Loss 才真正接管优化方向。这说明正则化是"辅助约束"而非"主驱动力"，权重必须小于重建 Loss 至少一个数量级。

3. **两阶段策略的必要性**：形状和颜色同步优化时，颜色梯度在形状还是球体时就开始"乱拉"顶点，造成相互干扰。先用 300 轮建立正确的几何骨架，再用 200 轮在稳定的形状基础上学颜色，两个子问题解耦后各自收敛质量明显更高。

4. **logit 空间参数化的工程价值**：将颜色参数 $\theta_c$ 定义在实数域，通过 $\text{sigmoid}$ 映射到 $[0,1]$，避免了颜色越界的截断问题，也保证了梯度在边界附近不会消失——这是可微优化中处理有界参数的标准工程范式。

5. **可微渲染的适用范围与局限**：本实验使用的是 Whitted-Style 软光栅化，只能优化顶点坐标和顶点颜色等显式参数。对于复杂的材质（BRDF）、光照环境和透明度，需要更高级的可微渲染框架（如 NeRF、3D Gaussian Splatting）才能有效建模，这也是当前计算机图形学研究的前沿方向。
