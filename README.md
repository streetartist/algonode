# AlgoNode

<div align="center">

🧮 **可视化数学建模平台**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

中文 | [English](README_EN.md)

</div>

---

## 📖 项目简介

AlgoNode 是一个基于 Flask 和 LiteGraph.js 构建的可视化节点编辑器，专为数学建模和算法设计而生。通过拖拽节点、连接数据流的方式，用户可以轻松构建复杂的数学模型，并一键导出为可运行的 Python 代码。

> *"The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise."*  
> *— Edsger W. Dijkstra*

## ✨ 功能特性

### 🎯 核心功能
- **可视化编辑器**：拖拽式节点操作，直观构建数学模型
- **实时运行**：在浏览器中直接运行模型并查看结果
- **代码导出**：自动生成可读性强的独立 Python 代码（基于 NumPy/SciPy）
- **模型保存/加载**：支持本地 JSON 格式保存和加载

### 📦 节点类型

#### 1. 优化模型 (Optimization)
| 节点 | 说明 |
|------|------|
| Linear/Integer Programming | 线性/整数规划 |
| Non-linear/Quadratic Programming | 非线性/二次规划 |
| Simulated Annealing | 模拟退火算法 |
| Genetic Algorithm | 遗传算法 |
| Neural Network Opt | 神经网络优化 |
| Dynamic Programming | 动态规划 (背包等) |
| Graph Algo (Dijkstra/MST/MaxFlow) | 图论算法 (最短路/生成树/最大流) |
| Combinatorial (TSP/VRP/Knapsack) | 组合优化 (旅行商/车辆路径/背包) |

> 线性/整数规划节点现在支持等式约束、变量上下界、最大化/最小化切换；新增 “Constraint Builder” 节点，可将 `1,2<=10;1,-1=3` 样式的文本转成 A/b 矩阵直接送入规划求解器。
- “Linear Model (Text)” 节点：直接编写目标、约束、边界、整数类型，生成 c/A/b/权重列表直接送入 LP/MIP 解析器，似于 LINGO/Matlab 的简明模型语法。
- 非线性规划节点强化：支持变量命名、上下界、文本格式等式约束，切换最大/最小，输出目标、状态与约束残差报告。

#### 2. 评价与决策 (Evaluation)
| 节点 | 说明 |
|------|------|
| AHP | 层次分析法 |
| TOPSIS | 优劣解距离法 |
| Fuzzy Evaluation | 模糊综合评价 |
| Grey Relational | 灰色关联分析 |
| RSR | 秩和比法 |
| Coupling Coordination | 耦合协调度 |
| BP NN Evaluation | BP神经网络评价 |
| PCA | 主成分分析 |

#### 3. 预测与时序 (Prediction)
| 节点 | 说明 |
|------|------|
| Linear/Logistic Regression | 线性/逻辑回归 |
| Polynomial Fitting | 多项式拟合 |
| Grey Prediction GM(1,1) | 灰色预测 |
| Time Series (ARIMA) | 时间序列分析 |
| Markov Chain | 马尔可夫链 |
| BP Neural Network | BP神经网络预测 |
| SVM/Random Forest/Decision Tree | 机器学习预测模型 |

#### 4. 统计与分析 (Statistics)
| 节点 | 说明 |
|------|------|
| Hypothesis Testing | 假设检验 (T检验/卡方/ANOVA) |
| Correlation Analysis | 相关性分析 |
| Discriminant Analysis | 判别分析 |
| Parameter Estimation | 参数估计 |

#### 5. 数学与仿真 (Math & Simulation)
| 节点 | 说明 |
|------|------|
| ODE Solver | 常微分方程数值解 |
| Monte Carlo | 蒙特卡洛模拟 |
| Numerical Integration | 数值积分 |
| Root Finding | 根查找 |
| Matrix Operations | 矩阵运算 (乘法/逆/特征值/解方程) |
| FFT | 快速傅里叶变换 |

#### 6. 数据与可视化 (Data & Viz)
| 节点 | 说明 |
|------|------|
| Data Processing | 归一化/分割/离散化/滤波 |
| Data Loading | 读取 CSV/Excel |
| 2D Plots | 折线图/散点图/直方图/箱线图 |
| 3D/Advanced Plots | 热力图/3D曲面图 |

#### 7. 扩展功能
| 节点 | 说明 |
|------|------|
| Custom Python Script | 自定义 Python 代码节点 |
| Subgraph | 子图封装与复用 |

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 现代浏览器（Chrome、Firefox、Edge 等）

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/streetartist/algonode.git
   cd algonode
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动应用**
   ```bash
   python app.py
   ```

4. **打开浏览器**
   
   访问 `http://localhost:5000`

## 📘 使用指南

1. **添加节点**：从左侧节点库中选择节点，或使用搜索框快速定位
2. **连接节点**：拖动节点端口建立数据流连接
3. **配置参数**：双击节点编辑属性和参数
4. **运行模型**：点击"运行模型"按钮查看执行结果
5. **导出代码**：点击"导出 Python 代码"生成独立脚本
6. **保存/加载**：支持本地 JSON 文件的保存和加载

## 📂 项目结构

```
algonode/
├── app.py                 # Flask 后端及代码生成逻辑
├── requirements.txt       # Python 依赖
├── user_library.json      # 用户自定义节点库
├── templates/
│   └── index.html         # 主页面模板
├── static/
│   ├── css/
│   │   └── style.css      # 样式文件
│   └── js/
│       └── app.js         # LiteGraph 配置及节点定义
├── examples/              # 示例模型
│   ├── 1_linear_regression.json    # 线性回归示例
│   ├── 2_ahp_evaluation.json       # AHP层次分析示例
│   ├── 3_tsp_optimization.json     # TSP优化示例
│   ├── 4_integer_programming.json  # 整数规划示例
│   └── ...
└── output/                # 编译输出目录
```

## 📋 依赖库

| 库 | 用途 |
|----|------|
| Flask | Web 框架 |
| NumPy | 数值计算 |
| SciPy | 科学计算 |
| scikit-learn | 机器学习 |
| NetworkX | 图论算法 |
| statsmodels | 统计建模 |
| pandas | 数据处理 |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目基于 [GPL-3.0](LICENSE) 许可证开源。

## 👤 作者

**闻家贤** - 电子科技大学

- GitHub: [@streetartist](https://github.com/streetartist)
