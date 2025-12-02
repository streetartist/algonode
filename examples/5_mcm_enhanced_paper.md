# MCM Problem A: Stair Wear Analysis (Enhanced Version)

## 1. Introduction
This document presents an enhanced solution to the MCM Problem A (Stair Wear Analysis) using the advanced features of AlgoNode. Unlike previous iterations that relied on hardcoded, problem-specific nodes, this solution leverages **Custom Python Scripting** and **Subgraphs** to create a flexible, modular, and reusable model.

## 2. Problem Restatement
The goal is to analyze the wear patterns on stairs to determine:
1.  The location of peak wear (indicating the most used path).
2.  The offset of this peak from the geometric center of the stairs.
3.  The total traffic volume over the lifespan of the stairs.

## 3. Methodology

### 3.1 Data Input
We input the raw measurement data using standard Vector nodes:
*   **X (Position)**: Measurements taken at 0, 10, 20, 30, 40, 50, 60 cm.
*   **Y (Wear Depth)**: Corresponding wear depths in mm.
*   **Constants**: Stair Center (30cm), Age (10 years), Daily Traffic (500 people).

### 3.2 Modular Modeling with Subgraphs
To keep the main workspace clean and organized, we encapsulate the core analysis logic within a **Subgraph** named "Stair Wear Model". This subgraph acts as a "black box" that takes raw data and constants as input and outputs the calculated metrics.

**Subgraph Structure:**
1.  **Polynomial Fitting**: Inside the subgraph, we use a standard `Polynomial Fitting` node (Degree 2) to model the wear pattern $y = ax^2 + bx + c$. This captures the parabolic nature of foot traffic wear.
2.  **Custom Logic**: Instead of a specialized "Wear Analysis" node, we use a generic **Custom Python Script** node. This node receives the coefficients $[a, b, c]$ from the fitting node and the other constants.

### 3.3 Custom Python Logic
The core mathematical derivation is implemented directly in Python within the Custom Script node. This demonstrates the platform's extensibility.

**Mathematical Derivation:**
The peak of the parabola $y = ax^2 + bx + c$ occurs at:
$$ x_{peak} = -\frac{b}{2a} $$

The offset is simply:
$$ \text{Offset} = x_{peak} - \text{Center} $$

Total traffic is estimated as:
$$ \text{Total} = \text{Age} \times \text{Daily Traffic} \times 365 $$

**Python Implementation:**
```python
# Inputs: coeffs, center, age, traffic
# Outputs: peak, offset, total
try:
    a = coeffs[0]
    b = coeffs[1]
    
    # Peak location
    if abs(a) > 1e-10:
        peak = -b / (2 * a)
    else:
        peak = 0
        
    offset = peak - center
    total = age * traffic * 365
except Exception:
    peak = 0
    offset = 0
    total = 0
```

## 4. Results
The model outputs the following metrics:
*   **Final_Peak**: The calculated position of maximum wear.
*   **Final_Offset**: The deviation from the center line.
*   **Final_Total**: The estimated total number of people who have used the stairs.

## 5. Conclusion
By using Subgraphs and Custom Python Scripts, we have created a solution that is:
*   **Generic**: No specialized C++ or backend code was needed for the specific math.
*   **Readable**: The logic is visible and editable within the node graph.
*   **Reusable**: The "Stair Wear Model" subgraph can be saved and reused in other projects.

This approach demonstrates the power of AlgoNode for rapid prototyping and mathematical modeling in contest environments.
