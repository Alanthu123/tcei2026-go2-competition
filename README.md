# TCEI 2026 — 四足机器狗特种场景应用挑战赛

> **2026 无人系统具身智能算法挑战赛** 参赛代码仓库

## 项目概述

本项目实现"**大模型 + 四足机器狗**"的虚实协同系统，基于宇树 Go2 机器狗，完成以下任务：

| 任务 | 说明 | 分值 |
|------|------|------|
| 巡线出发 | 沿 100mm 黑色导引线自主行进至战术拐角区 | 15 分 |
| 导航穿越 | SLAM + 路径规划穿越复杂地形 | 20 分 |
| 到达物资点 | 大模型语义理解 → 导航至指定军用物资存储点 | 15 分 |
| 抓取物资 | YOLO 目标检测 + 机械臂运动规划 + 夹爪抓取 | 15 分 |
| 到达终点 | 自主导航到达战术终点 | 15 分 |
| 大模型识别 | 九格 FM9G4B-V 多模态大模型目标识别（加分项） | 20 分 |

## 代码架构

```
tcei2026_go2_competition/
├── ros_ws/src/                  # ROS1 工作空间源码
│   ├── go2_control/             # 底盘运动控制 (巡线 + 速度拆分)
│   ├── go2_slam/                # 激光雷达点云转扫描
│   ├── go2_nav/                 # 定位 + 路径规划 + 导航
│   ├── go_arm/                  # YOLO 目标检测 + 机械臂抓取
│   └── go2_scale/               # 大模型指令解析 + 任务分发
├── simulation/Source/           # Isaac Sim 仿真源码
│   ├── Go2/                     # Go2 机器狗仿真 (策略控制 + 运动规划)
│   └── JAKA/                    # JAKA 机械臂仿真 (RMPflow + TRRT*)
├── inference/Embodied/          # 九格大模型推理模块
└── scripts/                     # 仿真启动脚本
```

## 关键技术

| 功能 | 文件位置 | 核心方法 | 语言 |
|------|---------|---------|------|
| 循迹 | [ros_ws/src/go2_control/src/advanced_line_follower.cpp](ros_ws/src/go2_control/src/advanced_line_follower.cpp) | BEV透视 + HSV阈值 + 最小二乘直线拟合 + PID | C++ |
| 点云转换 | [ros_ws/src/go2_slam/src/point_scan.cpp](ros_ws/src/go2_slam/src/point_scan.cpp) | 高度滤波 + 极坐标分桶 + 多帧累积 + 空洞插值 | C++ |
| 定位 | [ros_ws/src/go2_nav/src/lidar_localization.cpp](ros_ws/src/go2_nav/src/lidar_localization.cpp) | 梯度掩膜 + 暴力搜索匹配 + 指数平滑 + 滑动平均 | C++ |
| 路径规划 | [ros_ws/src/go2_nav/launch/](ros_ws/src/go2_nav/launch/) | ROS move_base (A* + DWA) | launch/XML |
| 目标检测 | [ros_ws/src/go_arm/scripts/go2_yolov8.py](ros_ws/src/go_arm/scripts/go2_yolov8.py) | Ultralytics YOLOv8 + Canny + minAreaRect | Python |
| 坐标变换 | [ros_ws/src/go_arm/src/go2_yolov8.cpp](ros_ws/src/go_arm/src/go2_yolov8.cpp) | 相机投影逆变换 + 手眼标定 (R·P+T) + 四元数旋转 | C++ |
| 抓取执行 | [ros_ws/src/go_arm/src/go2_grasp.cpp](ros_ws/src/go_arm/src/go2_grasp.cpp) | 状态机 (Home→Approach→Descent→Grasp→Home) | C++ |
| 仿真运动规划 | [simulation/Source/Go2/go2_env.py](simulation/Source/Go2/go2_env.py) | RMPflow + TRRT* + CubicSpline + 轨迹跟踪 | Python |
| 大模型推理 | [ros_ws/src/go2_scale/scripts/go2_scale.py](ros_ws/src/go2_scale/scripts/go2_scale.py) | FM9G4B-V 多模态VLM + 自然语言→ROS指令 | Python |

## 系统启动

```bash
# 终端1: ROS Core
roscore

# 终端2: Isaac Sim 仿真
cd /root/EAICON && bash run_go2_sim.sh

# 终端3: SLAM + 导航 + 巡线
cd /root/Go2 && roslaunch go2_slam go2_bring.launch

# 终端4: 大模型推理
cd /root/Go2 && conda activate inference && rosrun go2_scale go2_scale.py

# 终端5: YOLO 检测
cd /root/Go2 && conda activate yolov8 && rosrun go_arm go2_yolov8.py
```

## 依赖环境

- ROS1 (Noetic)
- NVIDIA Isaac Sim 4.5
- Conda 环境: `inference` (transformers + torch + 九格 FM9G4B-V)
- Conda 环境: `yolov8` (ultralytics + opencv)
- move_base + AMCL + TEB Local Planner

## 比赛规则摘要

- 场地: 5m × 6m 模拟城市废墟结构化场景
- 机器人: 宇树 Go2（主办方提供）
- 时间限制: 10 分钟
- 全自主运行，禁止遥控
- 必须使用九格 FM9G4B-V 大模型（未使用 = 0 分）
- 满分 100 分（5 个任务各 15 分 + 大模型识别 20 分 + 加分项）

## 许可证

本项目为 2026 无人系统具身智能算法挑战赛专用代码，依照 GNU GPL 条款授权。
