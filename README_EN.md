# AlgoNode

<div align="center">

🧮 **Visual Mathematical Modeling Platform**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

[中文](README.md) | English

</div>

---

## 📖 Introduction

AlgoNode is a visual node-based editor for mathematical modeling and algorithm design, built with Flask and LiteGraph.js. Create complex mathematical models by dragging nodes and connecting data flows, then export to runnable Python code with one click.

> *"The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise."*
> *— Edsger W. Dijkstra*

## ✨ Features

### 🎯 Core Features

- **Visual Editor**: Drag-and-drop node operations for intuitive model building
- **Real-time Execution**: Run models directly in the browser and view results
- **Code Export**: Auto-generate readable, standalone Python code (based on NumPy/SciPy)
- **Save/Load Models**: Support local JSON format for saving and loading

### 📦 Node Types

#### 1. Optimization Models

| Node                              | Description                                   |
| --------------------------------- | --------------------------------------------- |
| Linear/Integer Programming        | Linear/Integer programming solver             |
| Non-linear/Quadratic Programming  | Non-linear/Quadratic programming solver       |
| Simulated Annealing               | Simulated annealing optimization              |
| Genetic Algorithm                 | Genetic algorithm optimization                |
| Neural Network Opt                | Neural network optimization                   |
| Dynamic Programming               | Dynamic programming (Knapsack, etc.)          |
| Graph Algo (Dijkstra/MST/MaxFlow) | Graph algorithms (Shortest Path/MST/MaxFlow)  |
| Combinatorial (TSP/VRP/Knapsack)  | Combinatorial optimization (TSP/VRP/Knapsack) |

#### 2. Evaluation & Decision

| Node                  | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| AHP                   | Analytic Hierarchy Process                                     |
| TOPSIS                | Technique for Order Preference by Similarity to Ideal Solution |
| Fuzzy Evaluation      | Fuzzy comprehensive evaluation                                 |
| Grey Relational       | Grey relational analysis                                       |
| RSR                   | Rank Sum Ratio                                                 |
| Coupling Coordination | Coupling coordination degree                                   |
| BP NN Evaluation      | BP neural network evaluation                                   |
| PCA                   | Principal Component Analysis                                   |

#### 3. Prediction & Time Series

| Node                            | Description                        |
| ------------------------------- | ---------------------------------- |
| Linear/Logistic Regression      | Linear/Logistic regression         |
| Polynomial Fitting              | Polynomial curve fitting           |
| Grey Prediction GM(1,1)         | Grey prediction model              |
| Time Series (ARIMA)             | Time series analysis               |
| Markov Chain                    | Markov chain model                 |
| BP Neural Network               | BP neural network prediction       |
| SVM/Random Forest/Decision Tree | Machine learning prediction models |

#### 4. Statistics & Analysis

| Node                  | Description                                  |
| --------------------- | -------------------------------------------- |
| Hypothesis Testing    | Hypothesis testing (T-Test/Chi-Square/ANOVA) |
| Correlation Analysis  | Correlation analysis                         |
| Discriminant Analysis | Discriminant analysis                        |
| Parameter Estimation  | Parameter estimation                         |

#### 5. Math & Simulation

| Node                  | Description                               |
| --------------------- | ----------------------------------------- |
| ODE Solver            | Ordinary differential equation solver     |
| Monte Carlo           | Monte Carlo simulation                    |
| Numerical Integration | Numerical integration                     |
| Root Finding          | Root finding                              |
| Matrix Operations     | Matrix ops (Multiply/Inverse/Eigen/Solve) |
| FFT                   | Fast Fourier Transform                    |

#### 6. Data & Visualization

| Node              | Description                       |
| ----------------- | --------------------------------- |
| Data Processing   | Normalize/Split/Discretize/Filter |
| Data Loading      | Load CSV/Excel                    |
| 2D Plots          | Line/Scatter/Histogram/Box Plot   |
| 3D/Advanced Plots | Heatmap/3D Surface Plot           |

#### 7. Extensions

| Node                 | Description                    |
| -------------------- | ------------------------------ |
| Custom Python Script | Custom Python code node        |
| Subgraph             | Subgraph encapsulation & reuse |

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Modern browser (Chrome, Firefox, Edge, etc.)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/streetartist/algonode.git
   cd algonode
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**

   ```bash
   python app.py
   ```
4. **Open browser**

   Navigate to `http://localhost:5000`

## 📘 User Guide

1. **Add Nodes**: Select nodes from the sidebar or use the search box
2. **Connect Nodes**: Drag node ports to establish data flow connections
3. **Configure Parameters**: Double-click nodes to edit properties
4. **Run Model**: Click "运行模型" (Run Model) to see execution results
5. **Export Code**: Click "导出 Python 代码" (Export Python Code) to generate standalone scripts
6. **Save/Load**: Support saving and loading local JSON files

## 📂 Project Structure

```
algonode/
├── app.py                 # Flask backend and code generation logic
├── requirements.txt       # Python dependencies
├── user_library.json      # User-defined node library
├── templates/
│   └── index.html         # Main page template
├── static/
│   ├── css/
│   │   └── style.css      # Stylesheet
│   └── js/
│       └── app.js         # LiteGraph config and node definitions
└── examples/              # Example models
    ├── 1_linear_regression.json    # Linear regression example
    ├── 2_ahp_evaluation.json       # AHP evaluation example
    ├── 3_tsp_optimization.json     # TSP optimization example
    ├── 4_integer_programming.json  # Integer programming example
    └── ...
```

## 📋 Dependencies

| Library      | Purpose              |
| ------------ | -------------------- |
| Flask        | Web framework        |
| NumPy        | Numerical computing  |
| SciPy        | Scientific computing |
| scikit-learn | Machine learning     |
| NetworkX     | Graph algorithms     |
| statsmodels  | Statistical modeling |
| pandas       | Data processing      |

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the [GPL-3.0 License](LICENSE).

## 👤 Author

**Jiaxian Wen** - University of Electronic Science and Technology of China

- GitHub: [@streetartist](https://github.com/streetartist)
