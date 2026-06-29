# 实验零：现代图形学开发环境搭建与万有引力粒子群仿真

## 一、实验背景与目的

现代计算机图形学的开发不仅依赖高效的算法，同样需要一套规范、可维护的工程环境。本次实验以"打通完整图形学开发链路"为目标，围绕以下四个核心环节展开：

1. **环境搭建**：使用新一代包管理器 `uv` 构建项目级隔离虚拟环境，避免全局污染；
2. **逻辑解耦**：采用 `src` 布局（Source Layout）对代码进行分层拆分，物理隔离配置、计算与视图层；
3. **GPU 并行计算**：通过 Taichi 框架将物理计算内核自动编译并部署至 GPU，驱动大规模粒子群仿真；
4. **版本管理**：使用 Git 对项目进行版本控制，并同步至远程代码平台。

通过本次实验，建立起高效、统一的图形学开发工作流，为后续核心算法的实现打下基础。

---

## 二、核心工具简介

| 工具 | 说明 |
|------|------|
| **Trae IDE** | 基于 VS Code 内核的集成开发环境，集成 AI 辅助代码补全与排错功能 |
| **uv** | 基于 Rust 编写的新一代高性能 Python 包与环境管理器，支持项目级虚拟环境隔离 |
| **Taichi** | 面向高性能图形学与并行计算的 Python 库，支持 JIT 编译并自动调度至 CUDA / Metal / Vulkan 等后端 |
| **Git** | 分布式版本控制系统，用于追踪代码变更历史，支持多人协作与版本回退 |

---

## 三、项目架构

本项目严格遵循 `src` 布局规范，目录结构如下：

```

CG-Lab/
├── src/
│   └── Work0/
│       ├── **init**.py      # 包标识，使 Work0 成为可导入模块
│       ├── config.py        # 参数配置层：集中管理所有仿真参数
│       ├── physics.py       # 计算层：定义粒子数据结构与 GPU 内核
│       └── main.py          # 视图层：渲染循环与用户交互
├── pyproject.toml           # 项目元数据与依赖声明
├── .gitignore               # 忽略 .venv、**pycache** 等本地文件
└── README.md

````

### 分层设计说明

- **config.py（配置层）**：集中存放粒子数量、引力强度、阻力系数、窗口分辨率等所有可调参数，修改参数时无需深入业务代码；
- **physics.py（计算层）**：定义显存数据结构（`pos`、`vel`），并通过 `@ti.kernel` 装饰器声明 GPU 并行内核，负责粒子初始化与每帧物理更新；
- **main.py（视图层）**：负责 Taichi GUI 的创建、鼠标交互的捕获以及每帧的渲染驱动，不包含任何物理计算逻辑。

这种分层结构的核心价值在于：**各层职责单一，修改某一层不会影响其他层**，符合软件工程的解耦原则。

---

## 四、核心代码逻辑说明

### 4.1 GPU 初始化

```python
# main.py
ti.init(arch=ti.gpu)
````

`ti.init(arch=ti.gpu)` 会在程序启动时自动探测当前设备的最优 GPU 后端，优先级依次为：CUDA → Metal → Vulkan → OpenGL，若均不可用则回退至 CPU。**此语句必须在所有 Taichi 操作之前执行。**

### 4.2 显存数据结构

```python
# physics.py
pos = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
```

`ti.Vector.field` 在 GPU 显存中开辟连续内存空间，存储每个粒子的二维位置与速度。与普通 Python 列表不同，这块内存由 GPU 直接管理，可被并行内核高效访问。

### 4.3 GPU 并行物理内核

```python
# physics.py
@ti.kernel
def update_particles(mouse_x: float, mouse_y: float):
    for i in range(NUM_PARTICLES):
        mouse_pos = ti.Vector([mouse_x, mouse_y])
        dir = mouse_pos - pos[i]
        dist = dir.norm()
        if dist > 0.05:
            vel[i] += dir.normalized() * GRAVITY_STRENGTH
        vel[i] *= DRAG_COEF
        pos[i] += vel[i]
        for j in ti.static(range(2)):
            if pos[i][j] < 0:
                pos[i][j] = 0.0
                vel[i][j] *= BOUNCE_COEF
            elif pos[i][j] > 1:
                pos[i][j] = 1.0
                vel[i][j] *= BOUNCE_COEF
```

`@ti.kernel` 将此函数标记为 GPU 内核。Taichi 编译器会将 `for i in range(NUM_PARTICLES)` 自动展开为数千个并行线程，每个线程独立计算一个粒子的：

1. **引力加速度**：计算粒子到鼠标的方向向量，距离大于阈值时施加沿该方向的引力；
2. **速度衰减（阻力）**：每帧速度乘以小于 1 的 `DRAG_COEF`，模拟空气阻力；
3. **位置更新**：速度叠加到位置；
4. **边界碰撞**：检测粒子是否越界，越界则归位并对速度取反乘以弹性系数。

### 4.4 渲染主循环

```python
# main.py
while gui.running:
    mouse_x, mouse_y = gui.get_cursor_pos()
    update_particles(mouse_x, mouse_y)
    gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
    gui.show()
```

每帧流程：**捕获鼠标位置 → 调用 GPU 内核更新物理状态 → 将显存数据读回 CPU（`to_numpy()`）→ 绘制圆形粒子 → 刷新窗口**。

---

## 五、运行方式

### 环境要求

* Python 3.12+
* uv 包管理器
* 支持 CUDA / Vulkan / OpenGL 的显卡（或回退 CPU 运行）

### 安装依赖

```bash
uv sync
```

### 启动仿真

```bash
uv run -m src.Work0.main
```

### 交互说明

程序启动后会弹出渲染窗口，**移动鼠标即可改变引力中心**，所有粒子将实时向鼠标位置加速聚拢，并在边界处发生弹性碰撞。

---

## 六、GPU 调用验证

程序启动时，终端会输出 Taichi 的后端信息，以此判断 GPU 是否成功接管计算：

| 终端输出                                        | 含义                  | 状态      |
| ------------------------------------------- | ------------------- | ------- |
| `[Taichi] Starting on architecture: cuda`   | 调用 NVIDIA 独立显卡      | ✅ 最理想   |
| `[Taichi] Starting on architecture: metal`  | 调用 Apple M 系列芯片 GPU | ✅ 极佳    |
| `[Taichi] Starting on architecture: vulkan` | 调用集成显卡（Vulkan）      | ✅ 良好    |
| `[Taichi] Starting on architecture: opengl` | 调用集成显卡（OpenGL）      | ✅ 可用    |
| `[Taichi] Starting on architecture: cpu`    | 未找到兼容 GPU，回退 CPU    | ⚠️ 帧率较低 |

本机实际输出截图：

> ![终端输出](./assets/work0_terminal.png)


## 七、效果演示

> ![粒子仿真演示](./assets/work0_demo.gif)

鼠标移动时，粒子群实时响应引力场变化，聚拢、散开并在边界弹射，GPU 并行计算保障了大规模粒子的流畅渲染。

---

## 八、实验总结

通过本次实验，完成了以下核心目标：

1. 掌握了 `uv` 的项目级虚拟环境管理机制，理解了依赖隔离的必要性；
2. 理解并实践了 `src` 布局的分层解耦思想，将配置、计算、视图三层物理分离；
3. 通过 Taichi 的 `@ti.kernel` 机制，实现了物理计算向 GPU 的自动卸载，验证了并行计算在图形学仿真中的性能优势；
4. 完成了从本地开发到 Git 版本管理再到远程仓库同步的完整工作流。

本次实验建立的工程规范与开发流程，将作为后续全部图形学实验的统一基础。