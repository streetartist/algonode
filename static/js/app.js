// Initialize LiteGraph
var graph = new LGraph();
var canvas = new LGraphCanvas("#graphcanvas", graph);

// Resize canvas to fill window
function resize() {
    var parent = document.querySelector(".canvas-wrap");
    // Subtract status bar height if needed, but since status bar is absolute overlay, we can use full clientHeight
    canvas.resize(parent.clientWidth, parent.clientHeight);
}
window.addEventListener("resize", resize);
setTimeout(resize, 100);

// ---------------------------
// Status Bar & Double Click Logic
// ---------------------------

// Update status bar on selection
canvas.onNodeSelected = function(node) {
    var statusBar = document.getElementById("status-bar");
    if (statusBar) {
        // Get human readable label if possible
        var label = node.type;
        // Find label in categories
        for (var i=0; i<nodeCategories.length; i++) {
            var found = nodeCategories[i].nodes.find(n => n.type === node.type);
            if (found) {
                label = found.label + " (" + node.type + ")";
                break;
            }
        }
        statusBar.innerText = "Selected: " + label;
    }
};

canvas.onNodeDeselected = function(node) {
    var statusBar = document.getElementById("status-bar");
    if (statusBar) statusBar.innerText = "Ready";
};

// ---------------------------
// Node Registration Helper
// ---------------------------
function registerNode(type, title, inputs, outputs, properties, widgetType) {
    function Node() {
        if (inputs) {
            inputs.forEach(i => this.addInput(i[0], i[1]));
        }
        if (outputs) {
            outputs.forEach(o => this.addOutput(o[0], o[1]));
        }
        this.properties = Object.assign({}, properties || {});
        this._widgets = {};
        
        // Auto-add widgets for ALL properties with proper closure
        var self = this;
        Object.keys(this.properties).forEach(function(key) {
            var val = self.properties[key];
            var wType = "text";
            if (typeof val === "number") wType = "number";
            if (typeof val === "boolean") wType = "toggle";
            self._widgets[key] = self.addWidget(wType, key, val, (function(k) {
                return function(v) { self.properties[k] = v; };
            })(key));
        });
    }
    Node.prototype.onConfigure = function() {
        var self = this;
        if (this._widgets) {
            Object.keys(this._widgets).forEach(function(key) {
                if (self._widgets[key] && self.properties.hasOwnProperty(key)) {
                    self._widgets[key].value = self.properties[key];
                }
            });
        }
    };
    Node.title = title;
    LiteGraph.registerNodeType(type, Node);
    return type;
}

// ---------------------------
// Custom Node Definitions
// ---------------------------

// --- Basic Tools ---
// Data: Vector
function VectorNode() {
    this.addOutput("Vector", "array");
    this.properties = { name: "X", values: "1, 2, 3" };
    var self = this;
    // Name widget (was removed previously) — reintroducing for clarity
    this.nameWidget = this.addWidget("text", "Name", this.properties.name, function(v) { self.properties.name = v; });
    this.valuesWidget = this.addWidget("text", "Values (csv)", this.properties.values, function(v) { self.properties.values = v; });
}
VectorNode.prototype.onConfigure = function() {
    if (this.nameWidget) this.nameWidget.value = this.properties.name;
    if (this.valuesWidget) this.valuesWidget.value = this.properties.values;
};
VectorNode.title = "Vector";
LiteGraph.registerNodeType("data/vector", VectorNode);

// Data: Matrix
function MatrixNode() {
    this.addOutput("Matrix", "matrix");
    this.properties = { name: "M", rows: "1, 0; 0, 1" };
    var self = this;
    // Name widget (was removed previously) — reintroducing
    this.nameWidget = this.addWidget("text", "Name", this.properties.name, function(v) { self.properties.name = v; });
    this.rowsWidget = this.addWidget("text", "Rows (; for new row)", this.properties.rows, function(v) { self.properties.rows = v; });
}
MatrixNode.prototype.onConfigure = function() {
    if (this.nameWidget) this.nameWidget.value = this.properties.name;
    if (this.rowsWidget) this.rowsWidget.value = this.properties.rows;
};
MatrixNode.title = "Matrix";
LiteGraph.registerNodeType("data/matrix", MatrixNode);

// Math: Constant
function ConstantNode() {
    this.addOutput("Value", "number");
    this.properties = { name: "c", value: 1.0 };
    var self = this;
    // Name widget (was removed previously) — reintroducing
    this.nameWidget = this.addWidget("text", "Name", this.properties.name, function(v) { self.properties.name = v; });
    this.valueWidget = this.addWidget("number", "Value", this.properties.value, function(v) { self.properties.value = v; });
}
ConstantNode.prototype.onConfigure = function() {
    if (this.nameWidget) this.nameWidget.value = this.properties.name;
    if (this.valueWidget) this.valueWidget.value = this.properties.value;
};
ConstantNode.title = "Constant";
LiteGraph.registerNodeType("math/constant", ConstantNode);

// Custom: Python Script
function CustomPythonNode() {
    this.addInput("in1", "");
    this.addOutput("out1", "");
    this.properties = {
        code: "# Write your python code here\n# Inputs are available as variables named in 'Inputs' property\n# Assign results to variables named in 'Outputs' property\nout1 = in1 * 2",
        inputs: "in1",
        outputs: "out1"
    };
    var self = this;
    
    // Name widget (reintroduced)
    this.properties.name = this.properties.name || "script";
    this.nameWidget = this.addWidget("text", "Name", this.properties.name, function(v) { self.properties.name = v; });

    this.addWidget("text", "Inputs (comma sep)", this.properties.inputs, function(v) {
        self.properties.inputs = v;
        self.updateSlots();
    });
    
    this.addWidget("text", "Outputs (comma sep)", this.properties.outputs, function(v) {
        self.properties.outputs = v;
        self.updateSlots();
    });
    
    // Button to open the code editor modal
    this.addWidget("button", "Edit Code", null, function() {
        openPythonEditor(self);
    });
}

CustomPythonNode.prototype.updateSlots = function() {
    var input_names = this.properties.inputs.split(",").map(s => s.trim()).filter(s => s);
    var output_names = this.properties.outputs.split(",").map(s => s.trim()).filter(s => s);
    
    // Update Inputs
    // Remove extra
    while (this.inputs && this.inputs.length > input_names.length) {
        this.removeInput(this.inputs.length - 1);
    }
    // Add/Rename
    for (var i = 0; i < input_names.length; i++) {
        if (this.inputs && this.inputs[i]) {
            this.inputs[i].name = input_names[i];
        } else {
            this.addInput(input_names[i], "");
        }
    }
    
    // Update Outputs
    // Remove extra
    while (this.outputs && this.outputs.length > output_names.length) {
        this.removeOutput(this.outputs.length - 1);
    }
    // Add/Rename
    for (var i = 0; i < output_names.length; i++) {
        if (this.outputs && this.outputs[i]) {
            this.outputs[i].name = output_names[i];
        } else {
            this.addOutput(output_names[i], "");
        }
    }
};

CustomPythonNode.prototype.onConfigure = function() {
    this.updateSlots();
    // Also update widget display values to match properties
    if (this.widgets) {
        var self = this;
        this.widgets.forEach(function(w) {
            if (w.name === "Inputs (comma sep)") {
                w.value = self.properties.inputs || "";
            } else if (w.name === "Outputs (comma sep)") {
                w.value = self.properties.outputs || "";
            } else if (w.name === "Name") {
                w.value = self.properties.name || "";
            }
        });
    }
};

CustomPythonNode.title = "Custom Python Script";
LiteGraph.registerNodeType("custom/python_script", CustomPythonNode);


// Math Ops
function createMathNode(name, op, title) {
    function Node() {
        this.addInput("A", "number,array,matrix");
        this.addInput("B", "number,array,matrix");
        this.addOutput("Result", "number,array,matrix");
    }
    Node.title = title;
    LiteGraph.registerNodeType("math/" + name, Node);
}
createMathNode("add", "+", "Add");
createMathNode("subtract", "-", "Subtract");
createMathNode("multiply", "*", "Multiply");
createMathNode("divide", "/", "Divide");
createMathNode("power", "**", "Power");
registerNode("math/linspace", "Linspace", [], [["Vector", "array"]], { start: 0, stop: 1, num: 50 }, "text");

// Matrix-specific operations
function MatMulNode() {
    this.addInput("A", "matrix");
    this.addInput("B", "matrix");
    this.addOutput("Result", "matrix");
}
MatMulNode.title = "Matrix Multiply (A@B)";
LiteGraph.registerNodeType("math/matmul", MatMulNode);

function TransposeNode() {
    this.addInput("Matrix", "matrix");
    this.addOutput("Transposed", "matrix");
}
TransposeNode.title = "Transpose";
LiteGraph.registerNodeType("math/transpose", TransposeNode);

function InverseNode() {
    this.addInput("Matrix", "matrix");
    this.addOutput("Inverse", "matrix");
}
InverseNode.title = "Matrix Inverse";
LiteGraph.registerNodeType("math/inverse", InverseNode);

function DeterminantNode() {
    this.addInput("Matrix", "matrix");
    this.addOutput("Det", "number");
}
DeterminantNode.title = "Determinant";
LiteGraph.registerNodeType("math/determinant", DeterminantNode);
registerNode("math/lu_decompose", "LU Decomposition", [["A", "matrix"]], [["P", "matrix"], ["L", "matrix"], ["U", "matrix"]], { permute_l: false }, "text");
registerNode("math/qr", "QR Decomposition", [["A", "matrix"]], [["Q", "matrix"], ["R", "matrix"]], { mode: "reduced" }, "text");
registerNode("math/svd", "SVD", [["A", "matrix"]], [["U", "matrix"], ["S", "array"], ["Vh", "matrix"]], { full_matrices: false }, "text");
registerNode("math/conv", "1D Convolution", [["x", "array"], ["h", "array"]], [["y", "array"]], { mode: "full" }, "text");

// Output
function OutputNode() {
    this.addInput("In", "");
    this.properties = { name: "result" };
    var self = this;
    // Name widget (reintroduced)
    this.nameWidget = this.addWidget("text", "Name", this.properties.name, function(v) { self.properties.name = v; });
}
OutputNode.prototype.onConfigure = function() {
    if (this.nameWidget) this.nameWidget.value = this.properties.name;
};
OutputNode.title = "Output";
LiteGraph.registerNodeType("io/output", OutputNode);


// --- 1. Top 10 Algorithms ---

// 1. Monte Carlo
registerNode("algo/monte_carlo", "Monte Carlo Simulation", [], [["Result", "number"]], { iterations: 10000, seed: 42 }, "text");

// 2. Data Fitting
registerNode("algo/interpolation", "Interpolation", [["X", "array"], ["Y", "array"]], [["Model", "model"]], { method: "linear" }, "text");
registerNode("algo/parameter_estimation", "Parameter Estimation", [["X", "array"], ["Y", "array"]], [["Params", "array"]], {}, "text");

// 3. Planning
registerNode("algo/linear_programming", "Linear Programming", [["c", "array"], ["A_ub", "matrix"], ["b_ub", "array"]], [["Solution", "array"]], {}, "text");
registerNode("algo/integer_programming", "Integer Programming", [["c", "array"], ["A_ub", "matrix"], ["b_ub", "array"]], [["Solution", "array"]], {}, "text");
registerNode("algo/quadratic_programming", "Quadratic Programming", [["Q", "matrix"], ["c", "array"]], [["Solution", "array"]], {}, "text");

// 4. Graph Theory
registerNode("algo/dijkstra", "Dijkstra (Shortest Path)", [["Graph", "matrix"]], [["Distances", "array"]], { start: 0 }, "text");
registerNode("algo/mst", "Minimum Spanning Tree", [["Graph", "matrix"]], [["Tree", "matrix"]], {}, "text");
registerNode("algo/max_flow", "Max Flow", [["Graph", "matrix"]], [["Flow", "number"]], { source: 0, sink: -1 }, "text");

// 5. General Purpose Algorithms
// 动态规划节点（带下拉选择）
function DynamicProgrammingNode() {
    this.addInput("Values", "array");
    this.addInput("Weights", "array");
    this.addOutput("Result", "number");
    this.addOutput("Selected", "array");
    this.properties = { capacity: 50, problem_type: "knapsack" };
    var self = this;
    this.addWidget("number", "背包容量", this.properties.capacity, function(v) { self.properties.capacity = v; });
    this.addWidget("combo", "问题类型", this.properties.problem_type, function(v) { self.properties.problem_type = v; }, { values: ["knapsack"] });
}
DynamicProgrammingNode.title = "Dynamic Programming (General)";
LiteGraph.registerNodeType("algo/dynamic_programming", DynamicProgrammingNode);

// 回溯节点（带下拉选择）
function BacktrackingNode() {
    this.addInput("Constraints", "matrix");
    this.addOutput("Solution", "array");
    this.properties = { n: 8, problem_type: "n_queens" };
    var self = this;
    this.addWidget("number", "参数n", this.properties.n, function(v) { self.properties.n = v; });
    this.addWidget("combo", "问题类型", this.properties.problem_type, function(v) { self.properties.problem_type = v; }, { values: ["n_queens", "subset_sum", "permutation"] });
}
BacktrackingNode.title = "Backtracking Search";
LiteGraph.registerNodeType("algo/backtracking", BacktrackingNode);

// 分治节点（带下拉选择）
function DivideConquerNode() {
    this.addInput("Data", "array");
    this.addOutput("Result", "array");
    this.properties = { operation: "sort" };
    var self = this;
    this.addWidget("combo", "操作类型", this.properties.operation, function(v) { self.properties.operation = v; }, { values: ["sort", "median", "max_subarray"] });
}
DivideConquerNode.title = "Divide & Conquer";
LiteGraph.registerNodeType("algo/divide_conquer", DivideConquerNode);

// 6. Heuristic Optimization
registerNode("algo/simulated_annealing", "Simulated Annealing", [["Init", "array"]], [["Best", "array"]], { temp: 100, cooling: 0.95 }, "text");
registerNode("algo/genetic_algorithm", "Genetic Algorithm", [["Init", "array"]], [["Best", "array"]], { pop_size: 50, generations: 100 }, "text");
registerNode("algo/neural_network_opt", "Neural Network Opt", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], { hidden: "10,10" }, "text");

// 7. Grid & Exhaustive
registerNode("algo/grid_search", "Grid Search", [["Data", "matrix"], ["y", "array"]], [["BestParams", "array"]], { param_grid: "C:0.1,1,10" }, "text");
registerNode("algo/exhaustive_search", "Exhaustive Search", [["Data", "array"]], [["Best", "number"]], {}, "text");

// 8. Discretization
registerNode("algo/discretize", "Discretize Continuous", [["Data", "array"]], [["Discrete", "array"]], { bins: 10 }, "text");

// 9. Numerical Analysis
registerNode("algo/numerical_integration", "Numerical Integration", [["X", "array"], ["Y", "array"]], [["Area", "number"]], { method: "trapz" }, "text");
registerNode("algo/root_finding", "Root Finding", [["Coeffs", "array"]], [["Roots", "array"]], {}, "text");

// 10. Image Processing
registerNode("algo/image_filter", "Image Filter", [["Data", "matrix"]], [["Filtered", "matrix"]], { type: "blur", kernel: 3 }, "text");


// --- 2. Top 5 Models ---

// 2.1 Prediction
registerNode("model/bp_neural_network", "BP Neural Network", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], { hidden_layers: "10,10", max_iter: 500 }, "text");
registerNode("model/polynomial_fitting", "Polynomial Fitting", [["X", "array"], ["y", "array"]], [["Coeffs", "array"]], { degree: 2 }, "text");
registerNode("model/svm_predict", "SVM Prediction", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], { kernel: "rbf", C: 1.0 }, "text");
registerNode("model/grey_prediction", "Grey Prediction (GM(1,1))", [["Series", "array"]], [["Forecast", "array"]], { steps: 5 }, "text");
registerNode("model/time_series", "Time Series (ARIMA)", [["Series", "array"]], [["Forecast", "array"]], { p: 1, d: 1, q: 1, steps: 5 }, "text");
registerNode("model/markov_chain", "Markov Chain", [["TransMatrix", "matrix"], ["InitState", "array"]], [["NextState", "array"]], { steps: 1 }, "text");
registerNode("model/ridge_regression", "Ridge Regression", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], { alpha: 1.0, fit_intercept: true }, "text");
registerNode("model/lasso_regression", "Lasso Regression", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], { alpha: 1.0, fit_intercept: true }, "text");
registerNode("model/poly_features", "Polynomial Features", [["X", "matrix"]], [["X_poly", "matrix"]], { degree: 2, include_bias: true }, "text");

// 2.2 Evaluation
registerNode("eval/ahp", "AHP (Hierarchy)", [["Criteria", "matrix"]], [["Weights", "array"]], {}, "text");
registerNode("eval/topsis", "TOPSIS", [["Data", "matrix"], ["Weights", "array"]], [["Score", "array"]], { benefit: "1,1,1" }, "text");
registerNode("eval/fuzzy_eval", "Fuzzy Comprehensive Eval", [["R", "matrix"], ["W", "array"]], [["Result", "array"]], {}, "text");
registerNode("eval/grey_relational", "Grey Relational Analysis", [["Ref", "array"], ["Comp", "matrix"]], [["Rel", "array"]], { rho: 0.5 }, "text");
registerNode("eval/pca", "PCA", [["Data", "matrix"]], [["Components", "matrix"]], { n_components: 2 }, "text");
registerNode("eval/rsr", "Rank Sum Ratio", [["Data", "matrix"]], [["RSR", "array"]], {}, "text");
registerNode("eval/coupling", "Coupling Coordination", [["SysA", "array"], ["SysB", "array"]], [["Degree", "number"]], { alpha: 0.5 }, "text");
registerNode("eval/bp_eval", "BP NN Evaluation", [["X", "matrix"], ["y", "array"]], [["Score", "array"]], { hidden: "10" }, "text");

// 2.3 Classification
registerNode("class/kmeans", "K-Means Clustering", [["Data", "matrix"]], [["Labels", "array"]], { k: 3 }, "text");
registerNode("class/decision_tree", "Decision Tree", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], { max_depth: 5 }, "text");
registerNode("class/logistic_regression", "Logistic Regression", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], { C: 1.0 }, "text");
registerNode("class/random_forest", "Random Forest", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], { n_estimators: 100, max_depth: 5 }, "text");
registerNode("class/naive_bayes", "Naive Bayes", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], {}, "text");

// 2.4 Optimization (Some overlap with Top 10 Planning)
registerNode("opt/knapsack", "Knapsack Problem", [["Weights", "array"], ["Values", "array"]], [["Selected", "array"], ["MaxValue", "number"]], { capacity: 50 }, "text");
registerNode("opt/tsp", "TSP (Traveling Salesman)", [["DistMatrix", "matrix"]], [["Path", "array"], ["TotalDist", "number"]], {}, "text");
registerNode("opt/vrp", "Vehicle Routing", [["DistMatrix", "matrix"], ["Demands", "array"]], [["Routes", "array"]], { capacity: 100 }, "text");

// 2.5 Statistical Analysis
// Linear Regression Fit (Existing)
function LinearRegressionFit() {
    this.addInput("X", "array");
    this.addInput("y", "array");
    this.addOutput("Model", "model");
    this.properties = { name: "lr_model", fit_intercept: true };
    var self = this;
    // Name widget (reintroduced)
    this.nameWidget = this.addWidget("text", "Name", this.properties.name, function(v) { self.properties.name = v; });
    this.interceptWidget = this.addWidget("toggle", "Intercept", this.properties.fit_intercept, function(v) { self.properties.fit_intercept = v; });
}
LinearRegressionFit.prototype.onConfigure = function() {
    if (this.nameWidget) this.nameWidget.value = this.properties.name;
    if (this.interceptWidget) this.interceptWidget.value = this.properties.fit_intercept;
};
LinearRegressionFit.title = "Linear Regression Fit";
LiteGraph.registerNodeType("model/linear_regression_fit", LinearRegressionFit);

// Predict (Existing)
function PredictNode() {
    this.addInput("Model", "model");
    this.addInput("X", "array");
    this.addOutput("y_pred", "array");
    this.properties = { name: "y_pred" };
    var self = this;
    // Name widget (reintroduced)
    this.nameWidget = this.addWidget("text", "Name", this.properties.name, function(v) { self.properties.name = v; });
}
PredictNode.prototype.onConfigure = function() {
    if (this.nameWidget) this.nameWidget.value = this.properties.name;
};
PredictNode.title = "Predict";
LiteGraph.registerNodeType("model/predict", PredictNode);

// MSE (Existing)
function MSENode() {
    this.addInput("y_true", "array");
    this.addInput("y_pred", "array");
    this.addOutput("MSE", "number");
    this.properties = { name: "mse" };
    var self = this;
    // Name widget (reintroduced)
    this.nameWidget = this.addWidget("text", "Name", this.properties.name, function(v) { self.properties.name = v; });
}
MSENode.prototype.onConfigure = function() {
    if (this.nameWidget) this.nameWidget.value = this.properties.name;
};
MSENode.title = "MSE";
LiteGraph.registerNodeType("metrics/mse", MSENode);
registerNode("metrics/mae", "MAE", [["y_true", "array"], ["y_pred", "array"]], [["MAE", "number"]], {}, "text");
registerNode("metrics/rmse", "RMSE", [["y_true", "array"], ["y_pred", "array"]], [["RMSE", "number"]], {}, "text");
registerNode("metrics/r2", "R2 Score", [["y_true", "array"], ["y_pred", "array"]], [["R2", "number"]], {}, "text");
registerNode("metrics/accuracy", "Accuracy", [["y_true", "array"], ["y_pred", "array"]], [["Accuracy", "number"]], {}, "text");

registerNode("stat/correlation", "Correlation Analysis", [["X", "array"], ["Y", "array"]], [["Coeff", "number"]], {}, "text");
registerNode("stat/anova", "ANOVA", [["Group1", "array"], ["Group2", "array"]], [["F-stat", "number"], ["P-value", "number"]], {}, "text");
registerNode("stat/discriminant", "Discriminant Analysis", [["X", "matrix"], ["y", "array"]], [["Model", "model"]], {}, "text");

// --- New Nodes ---

// Data Preprocessing
registerNode("data/normalize", "Data Normalization", [["Data", "array"]], [["Normalized", "array"]], { method: "minmax" }, "text");
registerNode("data/split", "Train Test Split", [["X", "array"], ["y", "array"]], [["X_train", "array"], ["X_test", "array"], ["y_train", "array"], ["y_test", "array"]], { test_size: 0.2, random_state: 42 }, "text");
registerNode("data/load_csv", "Load CSV", [], [["Data", "matrix"]], { path: "data.csv", header: 0, encoding: "utf-8", usecols: "", output_format: "matrix" }, "text");
registerNode("data/load_excel", "Load Excel", [], [["Data", "matrix"]], { path: "data.xlsx", sheet: 0, usecols: "", output_format: "matrix" }, "text");
registerNode("data/load_excel_adv", "Load Excel+", [], [["Data", "matrix"]], { path: "data.xlsx", sheet: 0, header_rows: "1,3", data_start_row: "5", combine_mode: "code+name", city_column: "city", drop_empty_cols: true, output_format: "dataframe" }, "text");
registerNode("data/select_column", "Select Column", [["Data", "matrix"]], [["Column", "array"]], { selector: "0", mode: "index", as_array: true }, "text");
registerNode("data/filter_rows", "Filter Rows", [["Data", "matrix"]], [["Filtered", "matrix"]], { condition: "Total > 0", reset_index: true, output_format: "dataframe" }, "text");
registerNode("data/group_aggregate", "Group & Aggregate", [["Data", "matrix"]], [["Aggregated", "matrix"]], { group_by: "NOC", aggregations: "Gold:sum,Silver:sum", reset_index: true, flatten_columns: true, output_format: "dataframe" }, "text");
registerNode("data/describe", "Describe Data", [["Data", "matrix"]], [["Summary", "matrix"]], { include: "all", percentiles: "0.25,0.5,0.75" }, "text");
registerNode("data/pivot_table", "Pivot Table", [["Data", "matrix"]], [["Result", "matrix"]], { index: "", columns: "", values: "", aggfunc: "mean", fill_value: "", margins: false }, "text");
registerNode("data/conditional_column", "Conditional Column", [["Data", "matrix"]], [["Result", "matrix"]], { condition: "Gold > 0", true_value: 1, false_value: 0, output_column: "flag" }, "text");

// Enhanced Data Transformation Nodes
registerNode("data/rolling_window", "Rolling Window", [["Data", "matrix"]], [["Result", "matrix"]], { column: "Gold", window: 3, operation: "mean", groupby: "", min_periods: 1, output_column: "" }, "text");
registerNode("data/transform_column", "Transform Column", [["Data", "matrix"]], [["Result", "matrix"]], { column: "Gold", operation: "diff", periods: 1, fill_value: 0, decimals: 2, output_column: "" }, "text");
registerNode("data/merge_dataframes", "Merge DataFrames", [["Left", "matrix"], ["Right", "matrix"]], [["Merged", "matrix"]], { how: "inner", on: "", left_on: "", right_on: "" }, "text");
registerNode("data/time_features", "Time Features", [["Data", "matrix"]], [["Features", "matrix"]], { date_column: "Year", features: "year,month,dayofweek" }, "text");
registerNode("data/create_dummy", "Create Dummy Variables", [["Data", "matrix"]], [["Result", "matrix"]], { column: "NOC", mode: "onehot", value: "", output_column: "", prefix: "" }, "text");
registerNode("data/map_values", "Map Values", [["Data", "matrix"]], [["Result", "matrix"]], { column: "Year", mapping_dict: "{\"2024\": 329}", default_value: "", output_column: "mapped" }, "text");
registerNode("data/explode_column", "Explode Column", [["Data", "matrix"]], [["Result", "matrix"]], { column: "", output_column: "", ignore_index: true }, "text");
registerNode("data/expression", "Expression", [["Data", "matrix"]], [["Result", "matrix"]], { expression: "(A + B) / 2", output_column: "result" }, "text");
registerNode("data/weighted_score", "Weighted Normalize + Score", [["Data", "matrix"]], [["Scored", "matrix"]], { indicators: "X1:P:0.1,X2:N:0.1", normalize: "minmax", add_normalized_cols: true, score_column: "score", output_format: "dataframe" }, "text");
registerNode("data/indicator_dict", "Indicator Dictionary", [], [["Dict", "matrix"]], { indicators: "X1:P:0.1,X2:N:0.1", descriptions: "X1=desc;X2=desc" }, "text");

// Visualization
registerNode("viz/plot_line", "Line Plot", [["X", "array"], ["Y", "array"]], [], { title: "Line Plot" }, "text");
registerNode("viz/plot_scatter", "Scatter Plot", [["X", "array"], ["Y", "array"]], [], { title: "Scatter Plot" }, "text");

// Algo
registerNode("algo/ode_solver", "ODE Solver", [["y0", "array"], ["t", "array"]], [["y", "array"]], { dydt: "y" }, "text");

// Stat
registerNode("stat/ttest", "T-Test", [["Group1", "array"], ["Group2", "array"]], [["P-value", "number"]], {}, "text");
registerNode("stat/autocorr", "Auto-correlation", [["Series", "array"]], [["ACF", "array"], ["Lags", "array"]], { nlags: 40, demean: true }, "text");
registerNode("stat/pacf", "Partial Auto-corr", [["Series", "array"]], [["PACF", "array"]], { nlags: 20, method: "yw" }, "text");

// --- New Nodes (MATLAB/LINGO inspired) ---

// Math
registerNode("math/solve_linear", "Solve Linear System (Ax=b)", [["A", "matrix"], ["b", "array"]], [["x", "array"]], {}, "text");
registerNode("math/eigen", "Eigenvalues/Vectors", [["A", "matrix"]], [["Vals", "array"], ["Vecs", "matrix"]], {}, "text");
registerNode("math/fft", "FFT", [["X", "array"]], [["Spectrum", "array"]], {}, "text");

// Optimization
registerNode("algo/nonlinear_programming", "Non-linear Programming", [["x0", "array"]], [["Solution", "array"]], { objective: "x[0]**2 + x[1]**2", method: "SLSQP" }, "text");

// Statistics
registerNode("stat/chisquare", "Chi-Square Test", [["Observed", "array"], ["Expected", "array"]], [["P-value", "number"]], {}, "text");

// Signal
registerNode("signal/filter", "Signal Filter", [["Data", "array"]], [["Filtered", "array"]], { order: 4, cutoff: 0.1, btype: "low" }, "text");
registerNode("signal/resample", "Resample Signal", [["Data", "array"], ["fs_in", "number"], ["fs_out", "number"]], [["Resampled", "array"]], { axis: -1 }, "text");
registerNode("signal/stft", "Short-Time Fourier Transform", [["Data", "array"]], [["Freqs", "array"], ["Times", "array"], ["Z", "matrix"]], { fs: 1.0, nperseg: 256, noverlap: 128, window: "hann" }, "text");
registerNode("signal/bandpass_filter", "Bandpass Filter", [["Data", "array"]], [["Filtered", "array"]], { fs: 1.0, lowcut: 0.1, highcut: 0.5, order: 4, btype: "bandpass" }, "text");
registerNode("signal/xcorr", "Cross Correlation", [["X", "array"], ["Y", "array"]], [["Corr", "array"], ["Lags", "array"]], { mode: "full" }, "text");

// Control / Systems
registerNode("control/transfer_function", "Transfer Function", [["Numerator", "array"], ["Denominator", "array"]], [["System", "object"]], {}, "text");
registerNode("control/step_response", "Step Response", [["System", "object"]], [["t", "array"], ["y", "array"]], { T: "" }, "text");
registerNode("control/bode_plot", "Bode Plot", [["System", "object"]], [["w", "array"], ["mag", "array"], ["phase", "array"]], {}, "text");

// Visualization
registerNode("viz/plot_hist", "Histogram", [["Data", "array"]], [], { bins: 10, title: "Histogram" }, "text");
registerNode("viz/plot_box", "Box Plot", [["Data", "array"]], [], { title: "Box Plot" }, "text");
registerNode("viz/plot_heatmap", "Heatmap", [["Data", "matrix"]], [], { title: "Heatmap" }, "text");
registerNode("viz/plot_surface", "3D Surface Plot", [["X", "matrix"], ["Y", "matrix"], ["Z", "matrix"]], [], { title: "3D Surface" }, "text");


// ---------------------------
// Sidebar & Search Logic
// ---------------------------

const nodeCategories = [
    {
        name: "📊 数据与输入",
        nodes: [
            { type: "math/constant", label: "常量" },
            { type: "data/vector", label: "向量" },
            { type: "data/matrix", label: "矩阵" },
            { type: "data/load_csv", label: "读取 CSV" },
            { type: "data/load_excel", label: "读取 Excel" },
            { type: "data/load_excel_adv", label: "高级 Excel 读取" },
            { type: "data/select_column", label: "选择列" },
            { type: "data/filter_rows", label: "条件筛选" },
            { type: "data/group_aggregate", label: "分组聚合" },
            { type: "data/describe", label: "数据概览" },
            { type: "data/indicator_dict", label: "指标字典" },
            { type: "io/output", label: "输出节点" }
        ]
    },
    {
        name: "🧹 数据预处理",
        nodes: [
            { type: "data/normalize", label: "数据归一化" },
            { type: "data/split", label: "训练测试集分割" },
            { type: "data/rolling_window", label: "滚动窗口" },
            { type: "data/transform_column", label: "列变换" },
            { type: "data/merge_dataframes", label: "合并数据" },
            { type: "data/time_features", label: "时间特征" },
            { type: "data/create_dummy", label: "虚拟变量" },
            { type: "data/pivot_table", label: "透视表" },
            { type: "data/conditional_column", label: "条件生成列" },
            { type: "data/map_values", label: "值映射" },
            { type: "data/explode_column", label: "列展开" },
            { type: "data/expression", label: "表达式计算" },
            { type: "data/weighted_score", label: "指标加权归一化" }
        ]
    },
    {
        name: "📈 可视化",
        nodes: [
            { type: "viz/plot_line", label: "折线图" },
            { type: "viz/plot_scatter", label: "散点图" },
            { type: "viz/plot_hist", label: "直方图" },
            { type: "viz/plot_box", label: "箱线图" },
            { type: "viz/plot_heatmap", label: "热力图" },
            { type: "viz/plot_surface", label: "3D 曲面图" }
        ]
    },
    {
        name: "➕ 数学运算",
        nodes: [
            { type: "math/add", label: "加法" },
            { type: "math/subtract", label: "减法" },
            { type: "math/multiply", label: "乘法 (元素)" },
            { type: "math/divide", label: "除法" },
            { type: "math/power", label: "幂运算" },
            { type: "math/matmul", label: "矩阵乘法 (A@B)" },
            { type: "math/transpose", label: "矩阵转置" },
            { type: "math/inverse", label: "矩阵求逆" },
            { type: "math/determinant", label: "行列式" },
            { type: "math/solve_linear", label: "解线性方程组 (Ax=b)" },
            { type: "math/eigen", label: "特征值与特征向量" },
            { type: "math/fft", label: "快速傅里叶变换 (FFT)" },
            { type: "math/linspace", label: "等距序列 (linspace)" },
            { type: "math/lu_decompose", label: "LU 分解" },
            { type: "math/qr", label: "QR 分解" },
            { type: "math/svd", label: "奇异值分解 (SVD)" },
            { type: "math/conv", label: "一维卷积" }
        ]
    },
    {
        name: "📈 优化算法",
        nodes: [
            { type: "algo/linear_programming", label: "线性规划" },
            { type: "algo/integer_programming", label: "整数规划" },
            { type: "algo/quadratic_programming", label: "二次规划" },
            { type: "algo/nonlinear_programming", label: "非线性规划 (NLP)" },
            { type: "algo/simulated_annealing", label: "模拟退火" },
            { type: "algo/genetic_algorithm", label: "遗传算法" },
            { type: "opt/knapsack", label: "背包问题 (贪心)" },
            { type: "opt/tsp", label: "旅行商问题 (TSP)" },
            { type: "opt/vrp", label: "车辆路径问题 (VRP)" }
        ]
    },
    {
        name: "🔗 图论算法",
        nodes: [
            { type: "algo/dijkstra", label: "最短路径 (Dijkstra)" },
            { type: "algo/mst", label: "最小生成树" },
            { type: "algo/max_flow", label: "最大流" }
        ]
    },
    {
        name: "🤖 机器学习",
        nodes: [
            { type: "model/linear_regression_fit", label: "线性回归" },
            { type: "class/logistic_regression", label: "逻辑回归" },
            { type: "model/svm_predict", label: "支持向量机 (SVM)" },
            { type: "class/decision_tree", label: "决策树" },
            { type: "class/random_forest", label: "随机森林" },
            { type: "class/naive_bayes", label: "朴素贝叶斯" },
            { type: "class/kmeans", label: "K-Means 聚类" },
            { type: "eval/pca", label: "主成分分析 (PCA)" },
            { type: "model/ridge_regression", label: "岭回归" },
            { type: "model/lasso_regression", label: "Lasso 回归" },
            { type: "model/poly_features", label: "多项式特征" },
            { type: "model/bp_neural_network", label: "BP 神经网络" }
        ]
    },
    {
        name: "⚖️ 评价方法",
        nodes: [
            { type: "eval/ahp", label: "层次分析法 (AHP)" },
            { type: "eval/topsis", label: "TOPSIS 优劣解" },
            { type: "eval/fuzzy_eval", label: "模糊综合评价" },
            { type: "eval/grey_relational", label: "灰色关联分析" },
            { type: "eval/rsr", label: "秩和比综合评价" },
            { type: "eval/coupling", label: "耦合协调度" },
            { type: "eval/bp_eval", label: "BP 神经网络评价" }
        ]
    },
    {
        name: "🔮 预测方法",
        nodes: [
            { type: "model/grey_prediction", label: "灰色预测 GM(1,1)" },
            { type: "model/time_series", label: "时间序列 (ARIMA)" },
            { type: "model/markov_chain", label: "马尔可夫链" },
            { type: "algo/interpolation", label: "插值拟合" },
            { type: "model/polynomial_fitting", label: "多项式拟合" },
            { type: "model/predict", label: "通用预测" }
        ]
    },
    {
        name: "📐 数值分析",
        nodes: [
            { type: "algo/monte_carlo", label: "蒙特卡洛模拟" },
            { type: "algo/numerical_integration", label: "数值积分" },
            { type: "algo/root_finding", label: "方程求根" },
            { type: "algo/parameter_estimation", label: "参数估计" },
            { type: "algo/discretize", label: "连续离散化" },
            { type: "algo/ode_solver", label: "常微分方程求解" }
        ]
    },
    {
        name: "📡 信号与控制",
        nodes: [
            { type: "signal/filter", label: "信号滤波" },
            { type: "signal/resample", label: "信号重采样" },
            { type: "signal/stft", label: "短时傅里叶变换 (STFT)" },
            { type: "signal/bandpass_filter", label: "带通滤波" },
            { type: "signal/xcorr", label: "互相关" },
            { type: "control/transfer_function", label: "传递函数" },
            { type: "control/step_response", label: "阶跃响应" },
            { type: "control/bode_plot", label: "波特图" }
        ]
    },
    {
        name: "📊 统计分析",
        nodes: [
            { type: "stat/correlation", label: "相关分析" },
            { type: "stat/anova", label: "方差分析" },
            { type: "stat/ttest", label: "T检验" },
            { type: "stat/chisquare", label: "卡方检验" },
            { type: "stat/discriminant", label: "判别分析" },
            { type: "stat/autocorr", label: "自相关" },
            { type: "stat/pacf", label: "偏自相关" },
            { type: "metrics/mse", label: "均方误差 (MSE)" },
            { type: "metrics/mae", label: "平均绝对误差 (MAE)" },
            { type: "metrics/rmse", label: "均方根误差 (RMSE)" },
            { type: "metrics/r2", label: "决定系数 (R²)" },
            { type: "metrics/accuracy", label: "分类准确率" }
        ]
    },
    {
        name: "🔧 高级工具",
        nodes: [
            { type: "custom/python_script", label: "自定义 Python 代码" },
            { type: "graph/subgraph", label: "子图 (Subgraph)" },
            { type: "graph/input", label: "子图输入" },
            { type: "graph/output", label: "子图输出" },
            { type: "algo/dynamic_programming", label: "动态规划 (背包问题)" },
            { type: "algo/backtracking", label: "回溯搜索 (N皇后/子集/排列)" },
            { type: "algo/divide_conquer", label: "分治算法 (排序/中位数/最大子数组)" },
            { type: "algo/grid_search", label: "网格搜索" },
            { type: "algo/exhaustive_search", label: "穷举搜索" },
            { type: "algo/neural_network_opt", label: "神经网络优化" },
            { type: "algo/image_filter", label: "图像滤波" }
        ]
    }
];

// ---------------------------
// User Library System
// ---------------------------
var userLibrary = { nodes: [] };

function loadUserLibrary() {
    return fetch("/library")
        .then(res => res.json())
        .then(data => {
            if (data.ok) {
                userLibrary = data.library;
            }
            return userLibrary;
        })
        .catch(err => {
            console.error("Failed to load user library:", err);
            return userLibrary;
        });
}

function addNodeToLibrary(node) {
    // Open dialog to get name and description
    var name = prompt("给这个节点起个名字:", node.title || "我的节点");
    if (!name) return;
    
    var description = prompt("添加描述 (可选):", "");
    var category = prompt("分类 (默认: 我的节点):", "我的节点") || "我的节点";
    
    // Serialize the node data
    var nodeData = node.serialize();
    
    // For subgraphs, we need to include the subgraph data
    if (node.type === "graph/subgraph" && node.subgraph) {
        nodeData.subgraph = node.subgraph.serialize();
    }
    
    fetch("/library/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            node: nodeData,
            name: name,
            description: description,
            category: category
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.ok) {
            showToast(data.message);
            loadUserLibrary().then(() => renderSidebar());
        } else {
            showToast("保存失败: " + data.error, "error");
        }
    })
    .catch(err => {
        showToast("保存失败: " + err, "error");
    });
}

function deleteFromLibrary(libId, name) {
    if (!confirm(`确定要从库中删除 "${name}" 吗?`)) return;
    
    fetch(`/library/delete/${libId}`, { method: "DELETE" })
        .then(res => res.json())
        .then(data => {
            if (data.ok) {
                loadUserLibrary().then(() => renderSidebar());
            } else {
                showToast("删除失败: " + data.error, "error");
            }
        });
}

function instantiateLibraryNode(libraryEntry) {
    var nodeData = libraryEntry.node_data;
    var node;
    
    if (nodeData.type === "graph/subgraph") {
        // Create subgraph node
        node = LiteGraph.createNode("graph/subgraph");
        if (node) {
            node.pos = [150, 150];
            node.title = libraryEntry.name;
            
            // Restore inputs/outputs
            if (nodeData.inputs) {
                node.inputs = [];
                nodeData.inputs.forEach(inp => {
                    node.addInput(inp.name, inp.type);
                });
            }
            if (nodeData.outputs) {
                node.outputs = [];
                nodeData.outputs.forEach(out => {
                    node.addOutput(out.name, out.type);
                });
            }
            
            // Restore subgraph content
            if (nodeData.subgraph && node.subgraph) {
                node.subgraph.configure(nodeData.subgraph);
            }
            
            canvas.graph.add(node);

            // Ensure subgraph node has a name in properties and a visible Name widget
            node.properties = node.properties || {};
            node.properties.name = node.properties.name || libraryEntry.name || node.title || "subgraph";
            if (node.addWidget && (!node.widgets || !node.widgets.find(w => w.name === "Name"))) {
                (function(n){
                    n.nameWidget = n.addWidget("text", "Name", n.properties.name, function(v){ n.properties.name = v; });
                })(node);
            }
        }
    } else if (nodeData.type === "custom/python_script") {
        // Create custom python node
        node = LiteGraph.createNode("custom/python_script");
        if (node) {
            node.pos = [150, 150];
            node.title = libraryEntry.name;
            
            // Restore properties
            if (nodeData.properties) {
                node.properties = Object.assign({}, nodeData.properties);
                // Update slots based on inputs/outputs
                node.updateSlots();
                
                // Update widget display values to match restored properties
                if (node.widgets) {
                    node.widgets.forEach(function(w) {
                        if (w.name === "Inputs (comma sep)") {
                            w.value = node.properties.inputs || "";
                        } else if (w.name === "Outputs (comma sep)") {
                            w.value = node.properties.outputs || "";
                        }
                        // Update Name widget if present
                        if (w.name === "Name") {
                            w.value = node.properties.name || libraryEntry.name || node.title || "";
                        }
                    });
                    // If Name widget wasn't present, add it
                    if (node.addWidget && (!node.widgets || !node.widgets.find(w => w.name === "Name"))) {
                        (function(n){
                            n.nameWidget = n.addWidget("text", "Name", n.properties.name || libraryEntry.name || n.title || "", function(v){ n.properties.name = v; });
                        })(node);
                    }
                }
            }
            
            canvas.graph.add(node);
            // For generic manual nodes that carry a 'name' property but no widget, add Name widget
            if (nodeData.properties && nodeData.properties.name && node.addWidget && (!node.widgets || !node.widgets.find(w => w.name === "Name"))) {
                node.nameWidget = node.addWidget("text", "Name", node.properties.name, function(v){ node.properties.name = v; });
            }
        }
    } else {
        // Generic node creation
        node = LiteGraph.createNode(nodeData.type);
        if (node) {
            node.pos = [150, 150];
            if (nodeData.properties) {
                node.properties = Object.assign({}, nodeData.properties);
            }
            canvas.graph.add(node);
        }
    }
    
    return node;
}

function renderSidebar(filterText = "") {
    const container = document.getElementById("nodeList");
    container.innerHTML = "";
    
    const lowerFilter = filterText.toLowerCase();

    // First render built-in categories
    nodeCategories.forEach(cat => {
        // Filter nodes
        const visibleNodes = cat.nodes.filter(n => n.label.toLowerCase().includes(lowerFilter));
        
        // If searching, show all matching nodes expanded. If not, show categories.
        if (filterText && visibleNodes.length === 0) return;

        const catDiv = document.createElement("div");
        catDiv.className = "category";
        if (!filterText) {
            // Default collapsed state
            catDiv.classList.add("collapsed"); 
        }

        const header = document.createElement("div");
        header.className = "category-header";
        header.innerText = cat.name;
        header.onclick = () => {
            catDiv.classList.toggle("collapsed");
        };

        const content = document.createElement("div");
        content.className = "category-content";

        visibleNodes.forEach(nodeDef => {
            const btn = document.createElement("button");
            btn.className = "node-btn";
            btn.innerText = nodeDef.label;
            btn.onclick = () => {
                var node = LiteGraph.createNode(nodeDef.type);
                node.pos = [100, 100];
                // Use canvas.graph to ensure we add to the currently active graph (e.g. inside a subgraph)
                canvas.graph.add(node);
                // If user added a subgraph node via sidebar, ensure it has a Name widget
                if (node.type === "graph/subgraph") {
                    node.properties = node.properties || {};
                    node.properties.name = node.properties.name || node.title || "subgraph";
                    if (node.addWidget && (!node.widgets || !node.widgets.find(w => w.name === "Name"))) {
                        (function(n){
                            n.nameWidget = n.addWidget("text", "Name", n.properties.name, function(v){ n.properties.name = v; });
                        })(node);
                    }
                }
            };
            content.appendChild(btn);
        });

        catDiv.appendChild(header);
        catDiv.appendChild(content);
        container.appendChild(catDiv);
    });
    
    // Then render user library categories
    if (userLibrary && userLibrary.nodes && userLibrary.nodes.length > 0) {
        // Group by category
        var userCategories = {};
        userLibrary.nodes.forEach(entry => {
            var cat = entry.category || "我的节点";
            if (!userCategories[cat]) userCategories[cat] = [];
            userCategories[cat].push(entry);
        });
        
        Object.keys(userCategories).forEach(catName => {
            var entries = userCategories[catName];
            
            // Filter by search
            var visibleEntries = entries.filter(e => 
                e.name.toLowerCase().includes(lowerFilter) ||
                (e.description || "").toLowerCase().includes(lowerFilter)
            );
            
            if (filterText && visibleEntries.length === 0) return;
            
            const catDiv = document.createElement("div");
            catDiv.className = "category user-library-category";
            if (!filterText) {
                // Default collapsed state
                catDiv.classList.add("collapsed");
            }
            
            const header = document.createElement("div");
            header.className = "category-header";
            header.innerHTML = `📁 ${catName} <span class="user-lib-badge">用户库</span>`;
            header.onclick = () => {
                catDiv.classList.toggle("collapsed");
            };
            
            const content = document.createElement("div");
            content.className = "category-content";
            
            visibleEntries.forEach(entry => {
                const btnWrap = document.createElement("div");
                btnWrap.className = "user-node-wrap";
                
                const btn = document.createElement("button");
                btn.className = "node-btn user-lib-btn";
                btn.innerText = entry.name;
                btn.title = entry.description || "";
                btn.onclick = () => {
                    instantiateLibraryNode(entry);
                };
                
                const delBtn = document.createElement("button");
                delBtn.className = "node-btn-delete";
                delBtn.innerText = "×";
                delBtn.title = "从库中删除";
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    deleteFromLibrary(entry.lib_id, entry.name);
                };
                
                btnWrap.appendChild(btn);
                btnWrap.appendChild(delBtn);
                content.appendChild(btnWrap);
            });
            
            catDiv.appendChild(header);
            catDiv.appendChild(content);
            container.appendChild(catDiv);
        });
    }
}

// Initial Render - Load user library first, then render
loadUserLibrary().then(() => {
    renderSidebar();
});

// Add right-click menu to canvas for saving nodes to library
canvas.onShowNodePanel = function(node) {
    // This is called when right-clicking a node
    // We'll add our custom option via getExtraMenuOptions
};

// Patch nodes to add "Save to Library" option in context menu
var originalGetNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
LGraphCanvas.prototype.getNodeMenuOptions = function(node) {
    var options = originalGetNodeMenuOptions ? originalGetNodeMenuOptions.call(this, node) : [];
    
    // Add separator and our custom option for subgraph and custom python nodes
    if (node.type === "graph/subgraph" || node.type === "custom/python_script") {
        options.push(null); // separator
        options.push({
            content: "💾 保存到用户库",
            callback: function() {
                addNodeToLibrary(node);
            }
        });
    }
    
    return options;
};

// Patch Subgraph to ensure Name widget exists on load
// We need to patch after LiteGraph is fully ready
function patchSubgraphNode() {
    if (LiteGraph.Nodes["graph/subgraph"]) {
        const Subgraph = LiteGraph.Nodes["graph/subgraph"];
        const oldOnConfigure = Subgraph.prototype.onConfigure;
        Subgraph.prototype.onConfigure = function(o) {
            if (oldOnConfigure) oldOnConfigure.call(this, o);
            // Ensure a name property exists and widget reflects it
            try {
                this.properties = this.properties || {};
                if (!this.properties.name) this.properties.name = this.title || "subgraph";
                if (this.widgets && this.widgets.find) {
                    var w = this.widgets.find(x => x.name === "Name");
                    if (w) w.value = this.properties.name;
                }
            } catch(e) { /* defensive */ }
        };
        
        // Also patch the constructor to add widget on creation (for newly created subgraphs)
        const oldSubgraphInit = Subgraph.prototype.onAdded;
        Subgraph.prototype.onAdded = function() {
            if (oldSubgraphInit) oldSubgraphInit.call(this);
            // Add Name widget when a subgraph node is created
            try {
                this.properties = this.properties || {};
                this.properties.name = this.properties.name || this.title || "subgraph";
                if (this.addWidget && (!this.widgets || !this.widgets.find(w => w.name === "Name"))) {
                    var n = this;
                    n.nameWidget = n.addWidget("text", "Name", n.properties.name, function(v){ n.properties.name = v; });
                }
            } catch(e) { /* defensive */ }
        };
        return true;
    }
    return false;
}

// Try immediately, then retry after a short delay if needed
if (!patchSubgraphNode()) {
    setTimeout(patchSubgraphNode, 100);
}

// Also patch when graph is loaded/configured
var oldGraphConfigure = LGraph.prototype.configure;
LGraph.prototype.configure = function(data, keep_old) {
    var result = oldGraphConfigure.call(this, data, keep_old);
    // After graph is loaded, ensure subgraph nodes have name widget
    try {
        if (this._nodes && this._nodes.length) {
            this._nodes.forEach(function(n){
                try {
                    if (n.type === "graph/subgraph") {
                        n.properties = n.properties || {};
                        n.properties.name = n.properties.name || n.title || "subgraph";
                        if (n.addWidget && (!n.widgets || !n.widgets.find(w => w.name === "Name"))) {
                            n.nameWidget = n.addWidget("text", "Name", n.properties.name, function(v){ n.properties.name = v; });
                        }
                    }
                } catch(e) {}
            });
        }
    } catch(e) {}
    return result;
};

// Search Listener
document.getElementById("nodeSearch").addEventListener("input", (e) => {
    renderSidebar(e.target.value);
});


// ---------------------------
// CodeMirror Initialization
// ---------------------------
var exportEditor, pythonEditor;

window.addEventListener("load", function() {
    // Initialize Export Editor (Read-only)
    exportEditor = CodeMirror.fromTextArea(document.getElementById("codeOutput"), {
        mode: "python",
        theme: "monokai",
        lineNumbers: true,
        readOnly: true
    });

    // Initialize Python Script Editor (Editable)
    pythonEditor = CodeMirror.fromTextArea(document.getElementById("pythonCodeEditor"), {
        mode: "python",
        theme: "monokai",
        lineNumbers: true,
        indentUnit: 4,
        matchBrackets: true
    });
});

// ---------------------------
// Export / Import Logic
// ---------------------------

document.getElementById("btnExport").addEventListener("click", () => {
    var data = graph.serialize();
    fetch("/export", {
        method: "POST",
        body: JSON.stringify({ graph: data }),
        headers: { "Content-Type": "application/json" }
    })
    .then(res => res.json())
    .then(json => {
        if (json.ok) {
            // Update CodeMirror instead of textarea
            exportEditor.setValue(json.code);
            // Refresh to ensure proper rendering in modal
            setTimeout(() => exportEditor.refresh(), 10);
            document.getElementById("codeModal").classList.remove("hidden");
        } else {
            showToast("Error: " + json.error, "error");
        }
    })
    .catch(err => showToast("Network Error: " + err, "error"));
});

document.getElementById("closeModal").addEventListener("click", () => {
    document.getElementById("codeModal").classList.add("hidden");
});

document.getElementById("downloadPy").addEventListener("click", () => {
    // Get value from CodeMirror
    var content = exportEditor.getValue();
    var blob = new Blob([content], {type: "text/x-python"});
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "model.py";
    a.click();
});

// --- Run Model ---
document.getElementById("btnRun").addEventListener("click", () => {
    var data = graph.serialize();
    fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: data })
    })
    .then(response => response.json())
    .then(result => {
        const output = result.output || result.error;
        document.getElementById("runOutput").innerHTML = output;
        const modal = document.getElementById("resultModal");
        modal.classList.remove("hidden");
        modal.style.display = "flex";
    })
    .catch(err => {
        showToast("Error running model: " + err, "error");
    });
});

// --- Save/Load Local ---
document.getElementById("btnSaveLocal").addEventListener("click", () => {
    var data = graph.serialize();
    var str = JSON.stringify(data, null, 2);
    var blob = new Blob([str], {type: "application/json"});
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "graph.json";
    a.click();
});

document.getElementById("btnLoadLocal").addEventListener("click", () => {
    document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", (e) => {
    var file = e.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function(e) {
        try {
            var data = JSON.parse(e.target.result);
            graph.configure(data);
        } catch(err) {
            showToast("Invalid JSON file", "error");
        }
    };
    reader.readAsText(file);
});

// --- Result Modal Logic ---
document.querySelector("#resultModal .close").addEventListener("click", () => {
    const modal = document.getElementById("resultModal");
    modal.classList.add("hidden");
    modal.style.display = "none";
});

window.onclick = function(event) {
    var modal = document.getElementById("resultModal");
    if (event.target == modal) {
        modal.classList.add("hidden");
        modal.style.display = "none";
    }
    // Also handle codeModal if needed, though it might be handled elsewhere
    var codeModal = document.getElementById("codeModal");
    if (event.target == codeModal) {
        codeModal.classList.add("hidden");
    }
}

document.getElementById("btnSample").addEventListener("click", () => {
    graph.clear();
    var nodeX = LiteGraph.createNode("data/vector");
    nodeX.pos = [50, 50];
    nodeX.properties.name = "X_train";
    nodeX.properties.values = "1, 2, 3, 4, 5";
    graph.add(nodeX);
    nodeX.onConfigure();

    var nodeY = LiteGraph.createNode("data/vector");
    nodeY.pos = [50, 200];
    nodeY.properties.name = "y_train";
    nodeY.properties.values = "2.1, 3.9, 6.1, 8.0, 10.2";
    graph.add(nodeY);
    nodeY.onConfigure();

    var nodeFit = LiteGraph.createNode("model/linear_regression_fit");
    nodeFit.pos = [300, 100];
    graph.add(nodeFit);
    nodeFit.onConfigure();

    var nodeXTest = LiteGraph.createNode("data/vector");
    nodeXTest.pos = [300, 300];
    nodeXTest.properties.name = "X_test";
    nodeXTest.properties.values = "6, 7";
    graph.add(nodeXTest);
    nodeXTest.onConfigure();

    var nodePredict = LiteGraph.createNode("model/predict");
    nodePredict.pos = [600, 150];
    graph.add(nodePredict);
    nodePredict.onConfigure();

    var nodeOut = LiteGraph.createNode("io/output");
    nodeOut.pos = [850, 150];
    nodeOut.properties.name = "predictions";
    graph.add(nodeOut);
    nodeOut.onConfigure();

    nodeX.connect(0, nodeFit, 0);
    nodeY.connect(0, nodeFit, 1);
    nodeFit.connect(0, nodePredict, 0);
    nodeXTest.connect(0, nodePredict, 1);
    nodePredict.connect(0, nodeOut, 0);
});

graph.start();

// ---------------------------
// Python Code Editor Modal Logic
// ---------------------------
let currentEditingNode = null;
const pythonEditModal = document.getElementById("pythonEditModal");
// Note: pythonCodeEditor element is now managed by CodeMirror 'pythonEditor'

function openPythonEditor(node) {
    currentEditingNode = node;
    // Set value to CodeMirror
    pythonEditor.setValue(node.properties.code || "");
    pythonEditModal.classList.remove("hidden");
    pythonEditModal.style.display = "flex";
    // Refresh needed after modal becomes visible
    setTimeout(() => pythonEditor.refresh(), 10);
}

document.getElementById("btnSavePythonCode").addEventListener("click", () => {
    if (currentEditingNode) {
        // Get value from CodeMirror
        currentEditingNode.properties.code = pythonEditor.getValue();
        // Trigger a graph change if needed, though properties are updated directly
        currentEditingNode.setDirtyCanvas(true, true);
    }
    closePythonEditor();
});

document.getElementById("btnCancelPythonCode").addEventListener("click", () => {
    closePythonEditor();
});

document.getElementById("closePythonEdit").addEventListener("click", () => {
    closePythonEditor();
});

function closePythonEditor() {
    pythonEditModal.classList.add("hidden");
    pythonEditModal.style.display = "none";
    currentEditingNode = null;
}

// Close modal when clicking outside
window.addEventListener("click", function(event) {
    if (event.target == pythonEditModal) {
        closePythonEditor();
    }
    if (event.target == document.getElementById("aboutModal")) {
        closeAboutModal();
    }
});

// ---------------------------
// About Modal Logic
// ---------------------------
document.getElementById("btnAbout").addEventListener("click", () => {
    document.getElementById("aboutModal").classList.remove("hidden");
    document.getElementById("aboutModal").style.display = "flex";
});

document.getElementById("closeAbout").addEventListener("click", () => {
    closeAboutModal();
});

function closeAboutModal() {
    document.getElementById("aboutModal").classList.add("hidden");
    document.getElementById("aboutModal").style.display = "none";
}

// ---------------------------
// Node Market Logic
// ---------------------------
(function() {
    var marketModal = document.getElementById("marketModal");
    var uploadModal = document.getElementById("uploadModal");
    var btnMarket = document.getElementById("btnMarket");
    var btnPublish = document.getElementById("btnPublish");
    var closeMarket = document.getElementById("closeMarket");
    var closeUpload = document.getElementById("closeUpload");
    var filterBtns = document.querySelectorAll(".filter-btn");
    var marketList = document.getElementById("marketList");
    var btnDoUpload = document.getElementById("btnDoUpload");
    var uploadTypeSelect = document.getElementById("uploadType");
    var uploadTypeHint = document.getElementById("uploadTypeHint");
    var uploadTargetSelect = document.getElementById("uploadTarget");
    var uploadTargetGroup = document.getElementById("uploadTargetGroup");
    var centralServerInfo = document.getElementById("centralServerInfo");
    
    var currentFilter = "all";
    var allItems = [];
    var marketConfig = { central_server: null, mode: "client" };
    
    // Fetch market configuration
    function loadMarketConfig() {
        return fetch("/api/market/config")
            .then(res => res.json())
            .then(data => {
                if (data.ok) {
                    marketConfig = data;
                    updateUIForConfig();
                }
            })
            .catch(err => console.log("Failed to load market config"));
    }
    
    function updateUIForConfig() {
        // Update central server info display
        if (centralServerInfo) {
            // If this instance is the central server, show a different message
            if (marketConfig.is_central_server) {
                centralServerInfo.classList.remove("hidden", "disconnected");
                centralServerInfo.innerHTML = "✅ 中央服务器服务正常";
            } else if (marketConfig.central_server) {
                centralServerInfo.classList.remove("hidden");
                if (marketConfig.connected) {
                    centralServerInfo.classList.remove("disconnected");
                    centralServerInfo.innerHTML = "🌐 已连接中央服务器: " + marketConfig.central_server;
                } else {
                    centralServerInfo.classList.add("disconnected");
                    centralServerInfo.innerHTML = "⚠️ 无法连接中央服务器: " + marketConfig.central_server;
                }
            } else {
                centralServerInfo.classList.add("hidden");
            }
        }
        
        // Show/hide upload target option based on config
        if (uploadTargetGroup) {
            // Only allow selecting the remote target when a central server is configured AND connected
            if (marketConfig.central_server && !marketConfig.is_central_server && marketConfig.connected) {
                uploadTargetGroup.style.display = "block";
            } else {
                uploadTargetGroup.style.display = "none";
            }
        }
    }
    
    // Initialize config on page load
    loadMarketConfig();

    // Open market modal
    if (btnMarket) {
        btnMarket.onclick = function() {
            marketModal.classList.remove("hidden");
            loadMarketConfig().then(loadMarketItems);
        }
    }
    
    // Open upload modal
    if (btnPublish) {
        btnPublish.onclick = function() {
            loadMarketConfig();
            uploadModal.classList.remove("hidden");
            updateUploadTypeHint();
        }
    }
    
    if (closeMarket) {
        closeMarket.onclick = function() {
            marketModal.classList.add("hidden");
        }
    }
    
    if (closeUpload) {
        closeUpload.onclick = function() {
            uploadModal.classList.add("hidden");
        }
    }
    
    // Upload type hint update
    if (uploadTypeSelect) {
        uploadTypeSelect.onchange = updateUploadTypeHint;
    }
    
    function updateUploadTypeHint() {
        var type = uploadTypeSelect.value;
        var hints = {
            "project": "上传当前画布的所有内容作为完整项目。",
            "subgraph": "请先在画布上选中一个子图节点，将上传该子图。",
            "node": "请先在画布上选中一个自定义节点（如 Python Script），将上传该节点。"
        };
        uploadTypeHint.textContent = hints[type] || "";
    }

    // Filter buttons
    filterBtns.forEach(btn => {
        btn.onclick = function() {
            filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentFilter = btn.dataset.filter;
            renderMarketItems(allItems);
        }
    });

    function loadMarketItems() {
        marketList.innerHTML = '<div style="text-align: center; padding: 20px; color: #888;">加载中...</div>';
        fetch("/api/market/list")
            .then(res => res.json())
            .then(data => {
                if (data.ok) {
                    allItems = data.items;
                    renderMarketItems(allItems);
                } else {
                    marketList.innerHTML = '<div style="color: red;">加载失败</div>';
                }
            })
            .catch(err => {
                marketList.innerHTML = '<div style="color: red;">网络错误</div>';
            });
    }

    function renderMarketItems(items) {
        var filtered = items;
        if (currentFilter !== "all") {
            filtered = items.filter(item => item.type === currentFilter);
        }
        
        marketList.innerHTML = "";
        if (filtered.length === 0) {
            marketList.innerHTML = '<div style="text-align: center; padding: 20px; color: #888; grid-column: 1/-1;">暂无内容</div>';
            return;
        }
        filtered.forEach(item => {
            var div = document.createElement("div");
            div.className = "market-item";
            var date = new Date(item.timestamp * 1000).toLocaleDateString();
            var typeBadge = {
                "project": '<span class="type-badge project">项目</span>',
                "subgraph": '<span class="type-badge subgraph">子图</span>',
                "node": '<span class="type-badge node">节点</span>'
            }[item.type] || '<span class="type-badge project">项目</span>';
            
            // Source badge (local or remote)
            var sourceBadge = item.source === "remote" 
                ? '<span class="source-badge remote">云端</span>'
                : '<span class="source-badge local">本地</span>';
            
            var btnText = item.type === "project" ? "加载项目" : "导入到库";
            
            div.innerHTML = `
                <h4>${typeBadge}${item.name}${sourceBadge}</h4>
                <div class="meta">作者: ${item.author} | ${date}</div>
                <div class="desc">${item.description || "暂无描述"}</div>
                <button onclick="importMarketItem('${item.filename}', '${item.type}', '${item.source || "local"}')">${btnText}</button>
            `;
            marketList.appendChild(div);
        });
    }

    window.importMarketItem = function(filename, itemType, source) {
        var msg = itemType === "project" 
            ? "确定要加载这个项目吗？当前画布将被覆盖。"
            : "确定要导入这个内容到用户库吗？";
        if(!confirm(msg)) return;
        
        fetch("/api/market/import", {
            method: "POST",
            body: JSON.stringify({ filename: filename, source: source || "local" })
        })
        .then(res => res.json())
        .then(data => {
            if (data.ok) {
                if (data.type === "project") {
                    graph.configure(data.content);
                    showToast("项目加载成功！");
                    marketModal.classList.add("hidden");
                } else {
                    // subgraph or node imported to library
                    showToast("已导入到用户库！请刷新页面以在节点库中查看。", "info");
                    // Optionally reload user library
                    loadUserLibrary().then(() => renderSidebar());
                }
            } else {
                showToast("导入失败: " + data.error, "error");
            }
        })
        .catch(err => showToast("网络错误", "error"));
    };

    if (btnDoUpload) {
        btnDoUpload.onclick = function() {
            var name = document.getElementById("uploadName").value;
            var author = document.getElementById("uploadAuthor").value;
            var desc = document.getElementById("uploadDesc").value;
            var type = document.getElementById("uploadType").value;
            var target = uploadTargetSelect ? uploadTargetSelect.value : "local";
            
            if (!name) { showToast("请输入名称", "warning"); return; }
            
            var content = null;
            
            if (type === "project") {
                content = graph.serialize();
            } else if (type === "subgraph") {
                // Get selected node
                var selected = canvas.selected_nodes;
                var selectedArr = selected ? Object.values(selected) : [];
                if (selectedArr.length !== 1) {
                    showToast("请先在画布上选中一个子图节点", "warning"); return;
                }
                var node = selectedArr[0];
                if (node.type !== "graph/subgraph") {
                    showToast("选中的节点不是子图，请选择一个子图节点", "warning"); return;
                }
                // Serialize the subgraph node
                var nodeData = node.serialize();
                if (node.subgraph) {
                    nodeData.subgraph = node.subgraph.serialize();
                }
                content = nodeData;
            } else if (type === "node") {
                // Get selected node
                var selected = canvas.selected_nodes;
                var selectedArr = selected ? Object.values(selected) : [];
                if (selectedArr.length !== 1) {
                    showToast("请先在画布上选中一个节点", "warning"); return;
                }
                var node = selectedArr[0];
                if (node.type === "graph/subgraph") {
                    showToast("这是一个子图节点，请选择类型为'子图'进行上传", "warning"); return;
                }
                content = node.serialize();
            }
            
            btnDoUpload.innerText = "发布中...";
            btnDoUpload.disabled = true;
            
            fetch("/api/market/upload", {
                method: "POST",
                body: JSON.stringify({
                    name: name,
                    author: author,
                    description: desc,
                    type: type,
                    content: content,
                    target: target
                })
            })
            .then(res => res.json())
            .then(data => {
                if (data.ok) {
                    showToast("发布成功！");
                    uploadModal.classList.add("hidden");
                    // Clear form
                    document.getElementById("uploadName").value = "";
                    document.getElementById("uploadDesc").value = "";
                } else {
                    showToast("发布失败: " + data.error, "error");
                }
            })
            .catch(err => showToast("网络错误", "error"))
            .finally(() => {
                btnDoUpload.innerText = "确认发布";
                btnDoUpload.disabled = false;
            });
        }
    }

})();

// ---------------------------
// Toast Notification System
// ---------------------------
function showToast(message, type = "success", duration = 3000) {
    const container = document.getElementById("toast-container");
    if (!container) return;
    
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = "fadeOut 0.5s ease-out forwards";
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 500);
    }, duration);
}

// ---------------------------
// Auto-save & Shortcuts
// ---------------------------
(function() {
    const AUTOSAVE_KEY = "algonode_autosave";
    let lastSaveTime = 0;
    let hasUnsavedChanges = false;
    
    // Check for autosave on load
    const savedData = localStorage.getItem(AUTOSAVE_KEY);
    if (savedData) {
        try {
            const data = JSON.parse(savedData);
            // Only restore if graph is empty (default state)
            if (graph._nodes.length === 0) {
                // Simple check: if it's a fresh load, maybe just restore?
                // Or show a toast with action
                showToast("已恢复上次未保存的工作", "info");
                graph.configure(data);
            }
        } catch(e) {
            console.error("Failed to restore autosave", e);
        }
    }
    
    // Auto-save logic
    function autoSave() {
        if (!hasUnsavedChanges) return;
        
        const data = graph.serialize();
        localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(data));
        hasUnsavedChanges = false;
        lastSaveTime = Date.now();
        // showToast("自动保存成功", "info", 1000); // Optional: too many toasts might be annoying
    }
    
    // Track changes
    // LiteGraph doesn't have a generic 'change' event for everything, but we can hook into canvas events
    // A simple way is to check periodically if serialization changed, but that's heavy.
    // Better: hook into mouseup (drag end) and specific graph events.
    
    function markChanged() {
        hasUnsavedChanges = true;
    }
    
    canvas.onMouseUp = function(e) {
        if (e.which === 1) markChanged(); // Left click release usually means an action finished
        // Call original if needed, but LiteGraph assigns this directly usually.
        // To be safe, we should have preserved original, but LiteGraph's architecture is a bit raw.
        // Actually, LGraphCanvas.prototype.onMouseUp handles logic. We are overriding the instance method.
        // Let's use a safer approach: graph events.
    };
    
    // Graph events
    graph.onNodeAdded = markChanged;
    graph.onNodeRemoved = markChanged;
    graph.onNodeConnectionChange = markChanged;
    
    // Save every 30 seconds if changed
    setInterval(autoSave, 30000);
    
    // Save on page unload
    window.addEventListener("beforeunload", () => {
        if (hasUnsavedChanges) {
            const data = graph.serialize();
            localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(data));
        }
    });
    
    // Keyboard Shortcuts
    document.addEventListener("keydown", function(e) {
        // Ctrl + S : Save Local
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            document.getElementById("btnSaveLocal").click();
            showToast("已保存到本地文件");
        }
        
        // Ctrl + Enter : Run
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            document.getElementById("btnRun").click();
            showToast("开始运行模型...", "info");
        }
    });
    
})();

// ---------------------------
// Data Manager System
// ---------------------------
(function() {
    const btnDataManager = document.getElementById("btnDataManager");
    const dataModal = document.getElementById("dataModal");
    const closeDataModal = document.getElementById("closeDataModal");
    const dataFileList = document.getElementById("dataFileList");
    const btnSelectFile = document.getElementById("btnSelectFile");
    const dataFileUpload = document.getElementById("dataFileUpload");
    const uploadStatus = document.getElementById("uploadStatus");

    if (btnDataManager) {
        btnDataManager.onclick = function() {
            dataModal.classList.remove("hidden");
            loadDataFiles();
        };
    }

    if (closeDataModal) {
        closeDataModal.onclick = function() {
            dataModal.classList.add("hidden");
        };
    }
    
    // Close on outside click
    window.addEventListener("click", function(event) {
        if (event.target == dataModal) {
            dataModal.classList.add("hidden");
        }
    });

    function loadDataFiles() {
        dataFileList.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">加载中...</div>';
        
        fetch("/api/data/list")
            .then(res => res.json())
            .then(data => {
                if (data.ok) {
                    renderDataFiles(data.files);
                } else {
                    dataFileList.innerHTML = '<div style="padding: 20px; text-align: center; color: #f44336;">加载失败</div>';
                }
            })
            .catch(err => {
                dataFileList.innerHTML = '<div style="padding: 20px; text-align: center; color: #f44336;">网络错误</div>';
            });
    }

    function renderDataFiles(files) {
        if (files.length === 0) {
            dataFileList.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">暂无文件</div>';
            return;
        }

        dataFileList.innerHTML = "";
        files.forEach(file => {
            const item = document.createElement("div");
            item.className = "market-item"; // Reuse market item style
            item.style.flexDirection = "row";
            item.style.alignItems = "center";
            item.style.justifyContent = "space-between";
            item.style.marginBottom = "10px";
            
            const info = document.createElement("div");
            info.innerHTML = `
                <h4 style="margin: 0; color: #eee;">${file.name}</h4>
                <div class="meta" style="margin: 0;">Size: ${(file.size / 1024).toFixed(2)} KB</div>
            `;
            
            const actions = document.createElement("div");
            actions.style.display = "flex";
            actions.style.gap = "10px";
            
            const btnCopy = document.createElement("button");
            btnCopy.innerText = "📋 复制路径";
            btnCopy.className = "primary-btn";
            btnCopy.style.padding = "5px 10px";
            btnCopy.style.fontSize = "12px";
            btnCopy.onclick = () => {
                // Escape backslashes for Python string
                const pythonPath = file.path.replace(/\\/g, "/");
                navigator.clipboard.writeText(pythonPath).then(() => {
                    showToast("路径已复制到剪贴板");
                });
            };
            
            const btnDelete = document.createElement("button");
            btnDelete.innerText = "🗑️ 删除";
            btnDelete.style.padding = "5px 10px";
            btnDelete.style.fontSize = "12px";
            btnDelete.style.background = "#d32f2f";
            btnDelete.style.border = "none";
            btnDelete.style.color = "white";
            btnDelete.onclick = () => {
                if(confirm(`确定要删除 ${file.name} 吗?`)) {
                    fetch(`/api/data/delete/${file.name}`, { method: "DELETE" })
                        .then(res => res.json())
                        .then(data => {
                            if (data.ok) {
                                showToast("删除成功");
                                loadDataFiles();
                            } else {
                                showToast("删除失败: " + data.error, "error");
                            }
                        });
                }
            };
            
            actions.appendChild(btnCopy);
            actions.appendChild(btnDelete);
            
            item.appendChild(info);
            item.appendChild(actions);
            dataFileList.appendChild(item);
        });
    }

    if (btnSelectFile) {
        btnSelectFile.onclick = () => dataFileUpload.click();
    }

    if (dataFileUpload) {
        dataFileUpload.onchange = function() {
            if (this.files.length === 0) return;
            
            const file = this.files[0];
            const formData = new FormData();
            formData.append("file", file);
            
            uploadStatus.innerText = "上传中...";
            btnSelectFile.disabled = true;
            
            fetch("/api/data/upload", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.ok) {
                    showToast("上传成功");
                    loadDataFiles();
                } else {
                    showToast("上传失败: " + data.error, "error");
                }
            })
            .catch(err => showToast("网络错误", "error"))
            .finally(() => {
                uploadStatus.innerText = "支持 CSV, Excel 等格式";
                btnSelectFile.disabled = false;
                dataFileUpload.value = ""; // Reset
            });
        };
    }

})();
