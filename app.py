from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from typing import Dict, Any, List, Tuple, Set
import json
import os
import requests

# Load .env file if exists
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                # Don't override existing environment variables
                if key and key not in os.environ:
                    os.environ[key] = value

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "algonode_secret_key_default")

# -------------------------
# Admin Configuration
# -------------------------
ADMIN_PASSWORD = os.environ.get("ALGONODE_ADMIN_PASSWORD", "admin")

# -------------------------
# Central Server Configuration
# -------------------------
# Set ALGONODE_CENTRAL_SERVER environment variable to enable central server mode
# Example: ALGONODE_CENTRAL_SERVER=http://central-server.example.com:5000
# If not set, the market operates in local mode only
CENTRAL_SERVER_URL = os.environ.get("ALGONODE_CENTRAL_SERVER", "").rstrip("/")

# Set to "server" to run as a central server (accepts uploads from clients)
# Set to "client" or leave empty to run as a client (fetches from central server)
ALGONODE_MODE = os.environ.get("ALGONODE_MODE", "client")

# -------------------------
# Code Generation Utilities
# -------------------------

class Graph:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.nodes = {n["id"]: n for n in data.get("nodes", [])}
        self.links = data.get("links", [])
        self.in_edges: Dict[int, List[Tuple[int, int, int]]] = {}
        self.out_edges: Dict[int, List[Tuple[int, int, int]]] = {}
        for link in self.links:
            # LiteGraph links might have 6 elements (including type), so we take first 5
            if len(link) >= 5:
                _, src, src_slot, dst, dst_slot = link[:5]
                self.out_edges.setdefault(src, []).append((src, src_slot, dst))
                self.in_edges.setdefault(dst, []).append((src, src_slot, dst_slot))

    def topo_order(self) -> List[int]:
        indeg: Dict[int, int] = {nid: 0 for nid in self.nodes}
        for dst, in_list in self.in_edges.items():
            indeg[dst] = len(in_list)
        q = [nid for nid, d in indeg.items() if d == 0]
        order: List[int] = []
        while q:
            n = q.pop(0)
            order.append(n)
            for e in self.out_edges.get(n, []):
                _, _, dst = e
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    q.append(dst)
        if len(order) != len(self.nodes):
            order = list(self.nodes.keys())
        return order


def sanitize_name(name: str) -> str:
    import re
    # Keep the original case of the provided name while
    # normalizing spaces and removing invalid characters.
    # Previously this function lower-cased names which caused
    # the generated code to lose the original casing of `name`.
    # Now we preserve case so user-specified capitalization survives.
    s = name.strip().replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "", s)
    if not s:
        s = "var"
    if s[0].isdigit():
        s = f"v_{s}"
    return s


def default_var_name(node: Dict[str, Any]) -> str:
    """
    Generate a meaningful variable name for a node.

    Priority:
    1. Use node's custom title if provided
    2. Use node's name property if available
    3. Fall back to node type + id
    """
    # Check for custom title first
    title = node.get("title", "")
    props = node.get("properties", {}) or {}
    name_prop = props.get("name", "")

    # Use title if it's meaningful (not just the default type name)
    t = node.get("type", "node").split("/")[-1]
    if title and title != t and title.strip():
        base = sanitize_name(title)
    elif name_prop and name_prop.strip():
        base = sanitize_name(name_prop)
    else:
        base = sanitize_name(t)

    return f"{base}_{node['id']}"


def resolve_input_vars(node: Dict[str, Any], g: Graph, varmap: Dict[int, Any]) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    inputs = node.get("inputs") or []
    incoming = {dst_slot: (src, src_slot) for (src, src_slot, dst_slot) in g.in_edges.get(node["id"], [])}
    for i, inp in enumerate(inputs):
        inp_name = inp.get("name") or f"in{i}"
        if i in incoming:
            src, src_slot = incoming[i]
            # Handle multiple outputs: varmap[src] can be a dict {slot_index: var_name} or a single string
            val = varmap.get(src)
            if isinstance(val, dict):
                var_name = val.get(src_slot)
                # Fallback if slot not found (shouldn't happen if logic is correct)
                if var_name is None:
                    var_name = val.get(0, "None")
            else:
                var_name = val
            result.append((inp_name, var_name))
        else:
            result.append((inp_name, None))
    return result


def generate_scope(graph_data: Dict[str, Any], initial_varmap: Dict[int, Any] = None, readable: bool = True) -> Tuple[List[str], Dict[int, Any]]:
    """
    Generate code for a graph scope.

    Args:
        graph_data: The graph structure
        initial_varmap: Initial variable mapping (for subgraphs)
        readable: If True, add comments and better formatting
    """
    g = Graph(graph_data)
    order = g.topo_order()

    lines: List[str] = []
    varmap: Dict[int, Any] = initial_varmap.copy() if initial_varmap else {}
    current_category = None  # Track current node category for section comments

    def add(line: str):
        lines.append(line)

    def add_node_comment(node: Dict[str, Any]):
        """Add a descriptive comment for a node if readable mode is enabled."""
        if not readable:
            return
        ntype = node.get("type", "")
        title = node.get("title", "")
        desc = get_node_description(ntype)

        # Use title if available, otherwise use description
        if title and title != ntype.split("/")[-1]:
            comment = f"# {desc}: {title}"
        else:
            comment = f"# {desc}"
        add(comment)

    def add_section_comment(ntype: str):
        """Add section comment when category changes."""
        nonlocal current_category
        if not readable:
            return
        category = get_node_category(ntype)
        if category != current_category:
            current_category = category
            if lines:  # Add blank line before new section
                add("")
            add(f"# --- {category} ---")

    # --- Helper Generators ---

    def gen_constant(node, inputs):
        name = node.get("properties", {}).get("name") or default_var_name(node)
        v = node.get("properties", {}).get("value", 0)
        add(f"{sanitize_name(name)} = {v}")
        return sanitize_name(name)

    def gen_vector(node, inputs):
        name = node.get("properties", {}).get("name") or default_var_name(node)
        raw = node.get("properties", {}).get("values", "")
        try:
            values = [float(x) for x in raw.split(',') if x.strip() != '']
        except Exception:
            values = []
        add(f"{sanitize_name(name)} = np.array([{', '.join(str(x) for x in values)}])")
        return sanitize_name(name)

    def gen_matrix(node, inputs):
        name = node.get("properties", {}).get("name") or default_var_name(node)
        raw = node.get("properties", {}).get("rows", "")
        # Support both newline and semicolon as row delimiters
        raw = raw.replace(';', '\n')
        rows = []
        for line in raw.splitlines():
            parts = [p for p in line.split(',') if p.strip() != '']
            if parts:
                try:
                    rows.append([float(x) for x in parts])
                except Exception:
                    rows.append([])
        mat_literal = "; ".join(", ".join(str(x) for x in r) for r in rows)
        if rows:
            add(f"{sanitize_name(name)} = np.array([[{mat_literal.replace(';', '], [')}]])")
            add(f"{sanitize_name(name)} = np.array({sanitize_name(name)}.tolist())")
        else:
            add(f"{sanitize_name(name)} = np.zeros((0, 0))")
        return sanitize_name(name)

    def binop(op: str):
        def _f(node, inputs):
            a = inputs[0][1] if len(inputs) > 0 else None
            b = inputs[1][1] if len(inputs) > 1 else None
            name = default_var_name(node)
            a_expr = a if a is not None else "0"
            b_expr = b if b is not None else "0"
            add(f"{name} = ({a_expr}) {op} ({b_expr})")
            return name
        return _f

    def gen_power(node, inputs):
        a = inputs[0][1] if len(inputs) > 0 else None
        b = inputs[1][1] if len(inputs) > 1 else None
        name = default_var_name(node)
        a_expr = a if a is not None else "0"
        b_expr = b if b is not None else "1"
        add(f"{name} = np.power(({a_expr}), ({b_expr}))")
        return name

    # --- Matrix Operations ---

    def gen_matmul(node, inputs):
        a = inputs[0][1] if len(inputs) > 0 else None
        b = inputs[1][1] if len(inputs) > 1 else None
        name = default_var_name(node)
        a_expr = a if a is not None else "np.eye(2)"
        b_expr = b if b is not None else "np.eye(2)"
        add(f"# Matrix Multiplication (A @ B)")
        add(f"{name} = ({a_expr}) @ ({b_expr})")
        return name

    def gen_transpose(node, inputs):
        m = inputs[0][1] if len(inputs) > 0 else None
        name = default_var_name(node)
        m_expr = m if m is not None else "np.eye(2)"
        add(f"# Matrix Transpose")
        add(f"{name} = ({m_expr}).T")
        return name

    def gen_inverse(node, inputs):
        m = inputs[0][1] if len(inputs) > 0 else None
        name = default_var_name(node)
        m_expr = m if m is not None else "np.eye(2)"
        add(f"# Matrix Inverse")
        add(f"try:")
        add(f"    {name} = np.linalg.inv({m_expr})")
        add(f"except np.linalg.LinAlgError:")
        add(f"    {name} = np.full_like({m_expr}, np.nan)")
        return name

    def gen_determinant(node, inputs):
        m = inputs[0][1] if len(inputs) > 0 else None
        name = default_var_name(node)
        m_expr = m if m is not None else "np.eye(2)"
        add(f"# Matrix Determinant")
        add(f"{name} = np.linalg.det({m_expr})")
        return name

    def gen_linspace(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        start = float(props.get("start", 0))
        stop = float(props.get("stop", 1))
        num = int(props.get("num", 50))
        add(f"# Linspace sequence")
        add(f"{name} = np.linspace({start}, {stop}, {num})")
        return name

    def gen_lu_decompose(node, inputs):
        a = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        permute_l = bool(props.get("permute_l", False))
        add(f"# LU Decomposition")
        add(f"_A_{name} = np.asarray({a})")
        add(f"P_{name}, L_{name}, U_{name} = scipy.linalg.lu(_A_{name}, permute_l={permute_l})")
        add(f"{name}_P, {name}_L, {name}_U = P_{name}, L_{name}, U_{name}")
        return {0: f"{name}_P", 1: f"{name}_L", 2: f"{name}_U"}

    def gen_qr(node, inputs):
        a = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        mode = str(node.get("properties", {}).get("mode", "reduced")) or "reduced"
        add(f"# QR Decomposition")
        add(f"_A_{name} = np.asarray({a})")
        add(f"{name}_Q, {name}_R = np.linalg.qr(_A_{name}, mode={json.dumps(mode)})")
        return {0: f"{name}_Q", 1: f"{name}_R"}

    def gen_svd(node, inputs):
        a = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        full_matrices = bool(node.get("properties", {}).get("full_matrices", False))
        add(f"# Singular Value Decomposition")
        add(f"_A_{name} = np.asarray({a})")
        add(f"{name}_U, {name}_S, {name}_Vh = np.linalg.svd(_A_{name}, full_matrices={full_matrices})")
        return {0: f"{name}_U", 1: f"{name}_S", 2: f"{name}_Vh"}

    def gen_conv(node, inputs):
        x = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        h = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        mode = str(node.get("properties", {}).get("mode", "full")) or "full"
        add(f"# 1D Convolution")
        add(f"_x_{name} = np.asarray({x}).ravel()")
        add(f"_h_{name} = np.asarray({h}).ravel()")
        add(f"try:")
        add(f"    {name} = np.convolve(_x_{name}, _h_{name}, mode={json.dumps(mode)})")
        add(f"except Exception:")
        add(f"    {name} = np.array([])")
        return name

    # --- 1. Algo Implementations ---

    def gen_monte_carlo(node, inputs):
        name = default_var_name(node)
        iters = node.get("properties", {}).get("iterations", 10000)
        seed = node.get("properties", {}).get("seed", 42)
        add(f"# Monte Carlo Simulation (Estimate Pi)")
        add(f"np.random.seed({seed})")
        add(f"_pts_{name} = np.random.random(({iters}, 2))")
        add(f"{name} = 4 * np.sum(_pts_{name}[:,0]**2 + _pts_{name}[:,1]**2 <= 1) / {iters}")
        return name

    def gen_interpolation(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        Y = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        method = node.get("properties", {}).get("method", "linear")
        add(f"# Interpolation ({method})")
        add(f"{name} = scipy.interpolate.interp1d({X}, {Y}, kind='{method}', fill_value='extrapolate') if {X}.size > 1 else None")
        return name

    def gen_linprog(node, inputs):
        # Support equality constraints, bounds, labels, duals and max objective (via sign flip)
        c = inputs[0][1] if len(inputs) > 0 and inputs[0][1] is not None else "np.array([])"
        A_ub = inputs[1][1] if len(inputs) > 1 and inputs[1][1] is not None else "None"
        b_ub = inputs[2][1] if len(inputs) > 2 and inputs[2][1] is not None else "None"
        A_eq = inputs[3][1] if len(inputs) > 3 and inputs[3][1] is not None else "None"
        b_eq = inputs[4][1] if len(inputs) > 4 and inputs[4][1] is not None else "None"
        bounds_in = inputs[5][1] if len(inputs) > 5 and inputs[5][1] is not None else None
        ub_labels_in = inputs[6][1] if len(inputs) > 6 and inputs[6][1] is not None else None
        eq_labels_in = inputs[7][1] if len(inputs) > 7 and inputs[7][1] is not None else None
        var_labels_in = inputs[8][1] if len(inputs) > 8 and inputs[8][1] is not None else None
        sense = str(node.get("properties", {}).get("sense", "min")).lower()
        bounds_prop = node.get("properties", {}).get("bounds", "")

        # Parse bounds from property string "0,1;0,inf"
        bounds_list = []
        import re
        for line in re.split(r"[;\n]", bounds_prop):
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if not parts:
                continue
            if len(parts) >= 2:
                try:
                    lb = float(parts[0]) if parts[0].lower() not in ["none", "inf", "+inf"] else None
                except Exception:
                    lb = None
                try:
                    ub = float(parts[1]) if parts[1].lower() not in ["none", "inf", "+inf"] else None
                except Exception:
                    ub = None
                bounds_list.append((lb, ub))
        bounds_literal = bounds_list if bounds_list else None

        bounds_expr = bounds_in if bounds_in is not None else "None"

        name = default_var_name(node)

        # Step 1: Prepare input data
        add(f"")
        add(f"# Step 1: 准备输入数据 (Prepare input data)")
        add(f"_c_{name} = np.asarray({c}, dtype=float).ravel()")
        add(f"_Aub_{name} = (")
        add(f"    np.asarray({A_ub}, dtype=float)")
        add(f"    if {A_ub} is not None else None")
        add(f")")
        add(f"_bub_{name} = (")
        add(f"    np.asarray({b_ub}, dtype=float).ravel()")
        add(f"    if {b_ub} is not None else None")
        add(f")")
        add(f"_Aeq_{name} = (")
        add(f"    np.asarray({A_eq}, dtype=float)")
        add(f"    if {A_eq} is not None else None")
        add(f")")
        add(f"_beq_{name} = (")
        add(f"    np.asarray({b_eq}, dtype=float).ravel()")
        add(f"    if {b_eq} is not None else None")
        add(f")")

        # Step 2: Process bounds
        add(f"")
        add(f"# Step 2: 处理变量边界 (Process variable bounds)")
        add(f"_bounds_raw_{name} = (")
        add(f"    {bounds_expr} if {bounds_expr} is not None")
        add(f"    else {bounds_literal!r}")
        add(f")")
        add(f"if isinstance(_bounds_raw_{name}, np.ndarray):")
        add(f"    _bounds_raw_{name} = _bounds_raw_{name}.tolist()")
        add(f"_bounds_{name} = None")
        add(f"if _bounds_raw_{name}:")
        add(f"    _bounds_{name} = []")
        add(f"    for _lb, _ub in _bounds_raw_{name}:")
        add(f"        _lb_val = None if _lb in [None, -np.inf] else _lb")
        add(f"        _ub_val = None if _ub in [None, np.inf] else _ub")
        add(f"        _bounds_{name}.append((_lb_val, _ub_val))")

        # Step 3: Handle optimization sense
        add(f"")
        add(f"# Step 3: 设置优化方向 (Set optimization direction)")
        add(f"_sense_{name} = '{sense}'")
        add(f"_c_use_{name} = (")
        add(f"    _c_{name} if _sense_{name} != 'max'")
        add(f"    else -_c_{name}  # 最大化问题转换为最小化")
        add(f")")

        # Step 4: Prepare labels
        add(f"")
        add(f"# Step 4: 准备标签 (Prepare labels)")
        add(f"_ub_labels_{name} = list({ub_labels_in}) if {ub_labels_in} is not None else []")
        add(f"_eq_labels_{name} = list({eq_labels_in}) if {eq_labels_in} is not None else []")
        add(f"_var_labels_{name} = list({var_labels_in}) if {var_labels_in} is not None else []")

        # Step 5: Solve
        add(f"")
        add(f"# Step 5: 求解线性规划 (Solve linear programming)")
        add(f"res_{name} = (")
        add(f"    scipy.optimize.linprog(")
        add(f"        _c_use_{name},")
        add(f"        A_ub=_Aub_{name},")
        add(f"        b_ub=_bub_{name},")
        add(f"        A_eq=_Aeq_{name},")
        add(f"        b_eq=_beq_{name},")
        add(f"        bounds=_bounds_{name},")
        add(f"        method='highs'")
        add(f"    )")
        add(f"    if _c_{name}.size else None")
        add(f")")

        # Step 6: Extract results
        add(f"")
        add(f"# Step 6: 提取结果 (Extract results)")
        add(f"{name}_x = (")
        add(f"    res_{name}.x")
        add(f"    if res_{name} is not None and res_{name}.success")
        add(f"    else np.zeros_like(_c_{name})")
        add(f")")
        add(f"{name}_obj = (")
        add(f"    float(_c_{name}.dot({name}_x))")
        add(f"    if _c_{name}.size else 0.0")
        add(f")")
        add(f"{name}_status = (")
        add(f"    res_{name}.message")
        add(f"    if res_{name} is not None else 'no result'")
        add(f")")

        # Step 7: Extract dual values
        add(f"")
        add(f"# Step 7: 提取对偶值 (Extract dual values)")
        add(f"_dual_ub_vec_{name} = (")
        add(f"    getattr(getattr(res_{name}, 'ineqlin', None), 'marginals', None)")
        add(f"    if res_{name} is not None else None")
        add(f")")
        add(f"_dual_eq_vec_{name} = (")
        add(f"    getattr(getattr(res_{name}, 'eqlin', None), 'marginals', None)")
        add(f"    if res_{name} is not None else None")
        add(f")")
        add(f"_slack_ub_{name} = (")
        add(f"    getattr(getattr(res_{name}, 'ineqlin', None), 'slack', None)")
        add(f"    if res_{name} is not None else None")
        add(f")")
        add(f"_reduced_{name} = (")
        add(f"    getattr(res_{name}, 'reduced_cost', None)")
        add(f"    if res_{name} is not None else None")
        add(f")")

        # Step 8: Create named dictionaries
        add(f"")
        add(f"# Step 8: 创建命名字典 (Create named dictionaries)")
        add(f"{name}_dual_ub_named = None")
        add(f"if (isinstance(_dual_ub_vec_{name}, np.ndarray) and")
        add(f"        len(_ub_labels_{name}) == len(_dual_ub_vec_{name})):")
        add(f"    {name}_dual_ub_named = dict(zip(")
        add(f"        _ub_labels_{name},")
        add(f"        _dual_ub_vec_{name}.tolist()")
        add(f"    ))")
        add(f"")
        add(f"{name}_dual_eq_named = None")
        add(f"if (isinstance(_dual_eq_vec_{name}, np.ndarray) and")
        add(f"        len(_eq_labels_{name}) == len(_dual_eq_vec_{name})):")
        add(f"    {name}_dual_eq_named = dict(zip(")
        add(f"        _eq_labels_{name},")
        add(f"        _dual_eq_vec_{name}.tolist()")
        add(f"    ))")
        add(f"")
        add(f"{name}_reduced_named = None")
        add(f"if (isinstance(_reduced_{name}, np.ndarray) and")
        add(f"        len(_var_labels_{name}) == len(_reduced_{name})):")
        add(f"    {name}_reduced_named = dict(zip(")
        add(f"        _var_labels_{name},")
        add(f"        _reduced_{name}.tolist()")
        add(f"    ))")

        return {
            0: f"{name}_x",
            1: f"{name}_obj",
            2: f"{name}_status",
            3: f"_dual_ub_vec_{name}",
            4: f"_dual_eq_vec_{name}",
            5: f"_reduced_{name}",
            6: f"_slack_ub_{name}",
            7: f"{name}_dual_ub_named",
            8: f"{name}_dual_eq_named",
            9: f"{name}_reduced_named",
        }
    def gen_dijkstra(node, inputs):
        G_mat = inputs[0][1] if len(inputs) > 0 else "np.zeros((0,0))"
        start = node.get("properties", {}).get("start", 0)
        name = default_var_name(node)
        add(f"# Dijkstra Shortest Path (from node {start} to all)")
        add(f"G_{name} = nx.from_numpy_array({G_mat})")
        add(f"try:")
        add(f"    {name} = dict(nx.single_source_dijkstra_path_length(G_{name}, {start}))")
        add(f"    {name} = np.array([{name}.get(i, np.inf) for i in range(len({G_mat}))])")
        add(f"except Exception:")
        add(f"    {name} = np.array([])")
        return name

    def gen_mst(node, inputs):
        G_mat = inputs[0][1] if len(inputs) > 0 else "np.zeros((0,0))"
        name = default_var_name(node)
        add(f"# Minimum Spanning Tree")
        add(f"G_{name} = nx.from_numpy_array({G_mat})")
        add(f"T_{name} = nx.minimum_spanning_tree(G_{name})")
        add(f"{name} = nx.to_numpy_array(T_{name})")
        return name

    def gen_max_flow(node, inputs):
        G_mat = inputs[0][1] if len(inputs) > 0 else "np.zeros((0,0))"
        source = node.get("properties", {}).get("source", 0)
        sink = node.get("properties", {}).get("sink", -1)
        name = default_var_name(node)
        add(f"# Max Flow")
        add(f"G_{name} = nx.from_numpy_array({G_mat}, create_using=nx.DiGraph)")
        add(f"_sink_{name} = {sink} if {sink} >= 0 else len({G_mat}) - 1")
        add(f"try:")
        add(f"    {name} = nx.maximum_flow_value(G_{name}, {source}, _sink_{name})")
        add(f"except Exception:")
        add(f"    {name} = 0")
        return name

    def gen_simulated_annealing(node, inputs):
        init = inputs[0][1] if len(inputs) > 0 else "None"
        name = default_var_name(node)
        temp = node.get("properties", {}).get("temp", 100)
        cooling = node.get("properties", {}).get("cooling", 0.95)
        add(f"# Simulated Annealing (Optimization)")
        add(f"def obj_{name}(x): return np.sum(x**2)")
        add(f"_init_{name} = {init} if {init} is not None else np.random.random(5)")
        add(f"_bounds_{name} = [(-10, 10)] * len(_init_{name})")
        add(f"res_{name} = scipy.optimize.dual_annealing(obj_{name}, bounds=_bounds_{name}, initial_temp={temp})")
        add(f"{name} = res_{name}.x")
        return name

    def gen_genetic_algorithm(node, inputs):
        init = inputs[0][1] if len(inputs) > 0 else "None"
        name = default_var_name(node)
        pop_size = node.get("properties", {}).get("pop_size", 50)
        gens = node.get("properties", {}).get("generations", 100)
        add(f"# Genetic Algorithm (Differential Evolution)")
        add(f"def obj_{name}(x): return np.sum((x-1)**2)")
        add(f"_init_{name} = {init} if {init} is not None else None")
        add(f"_bounds_{name} = [(-5, 5)] * (len(_init_{name}) if _init_{name} is not None else 5)")
        add(f"res_{name} = scipy.optimize.differential_evolution(obj_{name}, bounds=_bounds_{name}, popsize={pop_size}, maxiter={gens})")
        add(f"{name} = res_{name}.x")
        return name

    def gen_numerical_integration(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        Y = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        method = node.get("properties", {}).get("method", "trapz")
        add(f"# Numerical Integration ({method})")
        add(f"if {X}.size and {Y}.size:")
        if method == "simpson":
            add(f"    from scipy.integrate import simpson")
            add(f"    {name} = simpson({Y}, x={X})")
        else:
            add(f"    {name} = np.trapz({Y}, {X})")
        add(f"else:")
        add(f"    {name} = 0")
        return name

    def gen_parameter_estimation(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        Y = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add(f"# Parameter Estimation (Linear Fit)")
        add(f"try:")
        add(f"    _coeffs_{name} = np.polyfit({X}, {Y}, 1)")
        add(f"    {name} = _coeffs_{name}")
        add(f"except Exception: {name} = np.array([])")
        return name

    def gen_integer_programming(node, inputs):
        # Mixed/Integer programming with bounds, equality, labels, gap and reduced costs
        c = inputs[0][1] if len(inputs) > 0 and inputs[0][1] is not None else "np.array([])"
        A_ub = inputs[1][1] if len(inputs) > 1 and inputs[1][1] is not None else "None"
        b_ub = inputs[2][1] if len(inputs) > 2 and inputs[2][1] is not None else "None"
        A_eq = inputs[3][1] if len(inputs) > 3 and inputs[3][1] is not None else "None"
        b_eq = inputs[4][1] if len(inputs) > 4 and inputs[4][1] is not None else "None"
        bounds_in = inputs[5][1] if len(inputs) > 5 and inputs[5][1] is not None else None
        integrality_in = inputs[6][1] if len(inputs) > 6 and inputs[6][1] is not None else None
        ub_labels_in = inputs[7][1] if len(inputs) > 7 and inputs[7][1] is not None else None
        eq_labels_in = inputs[8][1] if len(inputs) > 8 and inputs[8][1] is not None else None
        var_labels_in = inputs[9][1] if len(inputs) > 9 and inputs[9][1] is not None else None
        sense = str(node.get("properties", {}).get("sense", "min")).lower()
        bounds_prop = node.get("properties", {}).get("bounds", "")
        integrality_prop = node.get("properties", {}).get("integrality", "")

        # Parse bounds from property string
        import re
        bounds_list = []
        for line in re.split(r"[;\n]", bounds_prop):
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if not parts:
                continue
            if len(parts) >= 2:
                try:
                    lb = float(parts[0]) if parts[0].lower() not in ["none", "inf", "+inf"] else None
                except Exception:
                    lb = None
                try:
                    ub = float(parts[1]) if parts[1].lower() not in ["none", "inf", "+inf"] else None
                except Exception:
                    ub = None
                bounds_list.append((lb, ub))
        bounds_literal = bounds_list if bounds_list else None

        # Parse integrality string "1,0,2" -> vector
        integ_list = []
        for part in re.split(r"[;,]", integrality_prop):
            if part.strip():
                try:
                    integ_list.append(int(float(part.strip())))
                except Exception:
                    pass
        integ_literal = integ_list if integ_list else []
        bounds_expr = bounds_in if bounds_in is not None else "None"
        integ_expr = integrality_in if integrality_in is not None else "None"

        name = default_var_name(node)
        add(f"# Integer/Mixed-Integer Programming (SciPy milp with fallback)")
        add(f"_c_{name} = np.asarray({c}, dtype=float).ravel()")
        add(f"_Aub_{name} = np.asarray({A_ub}, dtype=float) if {A_ub} is not None else None")
        add(f"_bub_{name} = np.asarray({b_ub}, dtype=float).ravel() if {b_ub} is not None else None")
        add(f"_Aeq_{name} = np.asarray({A_eq}, dtype=float) if {A_eq} is not None else None")
        add(f"_beq_{name} = np.asarray({b_eq}, dtype=float).ravel() if {b_eq} is not None else None")
        add(f"_bounds_raw_{name} = {bounds_expr} if {bounds_expr} is not None else {bounds_literal!r}")
        add(f"if isinstance(_bounds_raw_{name}, np.ndarray): _bounds_raw_{name} = _bounds_raw_{name}.tolist()")
        add(f"_bounds_{name} = None")
        add(f"if _bounds_raw_{name}:")
        add(f"    _bounds_{name} = []")
        add(f"    for _lb, _ub in _bounds_raw_{name}:")
        add(f"        _lb_val = None if _lb in [None, -np.inf] else _lb")
        add(f"        _ub_val = None if _ub in [None, np.inf] else _ub")
        add(f"        _bounds_{name}.append((_lb_val, _ub_val))")
        add(f"_integrality_{name} = np.asarray({integ_expr} if {integ_expr} is not None else {integ_literal!r}, dtype=int) if _c_{name}.size else np.array([], dtype=int)")
        add(f"_ub_labels_{name} = list({ub_labels_in}) if {ub_labels_in} is not None else []")
        add(f"_eq_labels_{name} = list({eq_labels_in}) if {eq_labels_in} is not None else []")
        add(f"_var_labels_{name} = list({var_labels_in}) if {var_labels_in} is not None else []")
        add(f"_sense_{name} = '{sense}'")
        add(f"_c_use_{name} = _c_{name} if _sense_{name} != 'max' else -_c_{name}")
        add(f"_constraints_{name} = []")
        add(f"if _Aub_{name} is not None and _bub_{name} is not None:")
        add(f"    _constraints_{name}.append(scipy.optimize.LinearConstraint(_Aub_{name}, -np.inf, _bub_{name}))")
        add(f"if _Aeq_{name} is not None and _beq_{name} is not None:")
        add(f"    _constraints_{name}.append(scipy.optimize.LinearConstraint(_Aeq_{name}, _beq_{name}, _beq_{name}))")
        add(f"_status_note_{name} = ''")
        add(f"if hasattr(scipy.optimize, 'milp') and _c_{name}.size:")
        add(f"    res_{name} = scipy.optimize.milp(_c_use_{name}, integrality=_integrality_{name}, bounds=_bounds_{name}, constraints=_constraints_{name})")
        add(f"    _status_note_{name} = ''")
        add(f"else:")
        add(f"    res_{name} = scipy.optimize.linprog(_c_use_{name}, A_ub=_Aub_{name}, b_ub=_bub_{name}, A_eq=_Aeq_{name}, b_eq=_beq_{name}, bounds=_bounds_{name}, method='highs') if _c_{name}.size else None")
        add(f"    _status_note_{name} = ' (relaxed linprog fallback)'")
        add(f"if res_{name} is not None and getattr(res_{name}, 'success', False):")
        add(f"    _x_{name} = res_{name}.x")
        add(f"    if _integrality_{name}.size:")
        add(f"        _x_{name} = np.array(_x_{name}, copy=True)")
        add(f"        for _i, _itype in enumerate(_integrality_{name}):")
        add(f"            if _i < len(_x_{name}) and _itype:")
        add(f"                if _itype == 2:")
        add(f"                    _x_{name}[_i] = 1 if _x_{name}[_i] >= 0.5 else 0")
        add(f"                else:")
        add(f"                    _x_{name}[_i] = np.round(_x_{name}[_i])")
        add(f"else:")
        add(f"    _x_{name} = np.zeros_like(_c_{name})")
        add(f"{name}_obj = float(_c_{name}.dot(_x_{name})) if _c_{name}.size else 0.0")
        add(f"{name}_status = (res_{name}.message if res_{name} is not None else 'no result') + _status_note_{name}")
        add(f"{name}_gap = getattr(res_{name}, 'mip_gap', None) if res_{name} is not None else None")
        add(f"_reduced_{name} = getattr(res_{name}, 'reduced_cost', None) if res_{name} is not None else None")
        add(f"{name}_reduced_named = None")
        add(f"if isinstance(_reduced_{name}, np.ndarray) and len(_var_labels_{name}) == len(_reduced_{name}):")
        add(f"    {name}_reduced_named = dict(zip(_var_labels_{name}, _reduced_{name}.tolist()))")
        add(f"_dual_ub_vec_{name} = getattr(getattr(res_{name}, 'ineqlin', None), 'marginals', None) if res_{name} is not None else None")
        add(f"_dual_eq_vec_{name} = getattr(getattr(res_{name}, 'eqlin', None), 'marginals', None) if res_{name} is not None else None")
        add(f"{name}_dual_ub_named = None")
        add(f"if isinstance(_dual_ub_vec_{name}, np.ndarray) and len(_ub_labels_{name}) == len(_dual_ub_vec_{name}):")
        add(f"    {name}_dual_ub_named = dict(zip(_ub_labels_{name}, _dual_ub_vec_{name}.tolist()))")
        add(f"{name}_dual_eq_named = None")
        add(f"if isinstance(_dual_eq_vec_{name}, np.ndarray) and len(_eq_labels_{name}) == len(_dual_eq_vec_{name}):")
        add(f"    {name}_dual_eq_named = dict(zip(_eq_labels_{name}, _dual_eq_vec_{name}.tolist()))")
        return {
            0: f"_x_{name}",
            1: f"{name}_obj",
            2: f"{name}_status",
            3: f"{name}_gap",
            4: f"_reduced_{name}",
            5: f"{name}_reduced_named",
            6: f"_dual_ub_vec_{name}",
            7: f"_dual_eq_vec_{name}",
            8: f"{name}_dual_ub_named",
            9: f"{name}_dual_eq_named",
        }
    def gen_constraint_builder(node, inputs):
        raw = str(node.get("properties", {}).get("constraints", ""))
        name = default_var_name(node)
        import re
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        labels_ub = []
        labels_eq = []
        ub_count = 0
        eq_count = 0
        for line in re.split(r"[;\n]", raw):
            ln = line.strip()
            if not ln:
                continue
            label = None
            if ":" in ln:
                label_part, ln = ln.split(":", 1)
                if label_part.strip():
                    label = sanitize_name(label_part.strip())
                ln = ln.strip()
            m = re.match(r"(.+?)(<=|>=|=)(.+)")
            if not m:
                continue
            coeff_part, op, rhs_part = m.groups()
            coeff_tokens = [tok for tok in re.split(r"[ ,]+", coeff_part.strip()) if tok]
            try:
                coeffs = [float(tok) for tok in coeff_tokens]
                rhs_val = float(rhs_part.strip())
            except Exception:
                continue
            if op == ">=":
                coeffs = [-c for c in coeffs]
                rhs_val = -rhs_val
                A_ub.append(coeffs)
                b_ub.append(rhs_val)
                labels_ub.append(label or f"ub_{ub_count}")
                ub_count += 1
            elif op == "<=":
                A_ub.append(coeffs)
                b_ub.append(rhs_val)
                labels_ub.append(label or f"ub_{ub_count}")
                ub_count += 1
            else:
                A_eq.append(coeffs)
                b_eq.append(rhs_val)
                labels_eq.append(label or f"eq_{eq_count}")
                eq_count += 1
        aub_var = f"{name}_A_ub"
        bub_var = f"{name}_b_ub"
        aeq_var = f"{name}_A_eq"
        beq_var = f"{name}_b_eq"
        lab_ub_var = f"{name}_labels_ub"
        lab_eq_var = f"{name}_labels_eq"
        add(f"# Constraint builder from text (labels + <=, >=, =)")
        add(f"{aub_var} = np.array({A_ub!r}, dtype=float) if {bool(A_ub)} else None")
        add(f"{bub_var} = np.array({b_ub!r}, dtype=float) if {bool(b_ub)} else None")
        add(f"{aeq_var} = np.array({A_eq!r}, dtype=float) if {bool(A_eq)} else None")
        add(f"{beq_var} = np.array({b_eq!r}, dtype=float) if {bool(b_eq)} else None")
        add(f"{lab_ub_var} = {labels_ub!r}")
        add(f"{lab_eq_var} = {labels_eq!r}")
        return {0: aub_var, 1: bub_var, 2: aeq_var, 3: beq_var, 4: lab_ub_var, 5: lab_eq_var}

    def gen_linear_model_builder(node, inputs):
        # Parse a small LINGO-like text model into matrices/labels for LP/MIP solvers
        import re
        props = node.get("properties", {}) or {}
        vars_prop = str(props.get("variables", "") or "")
        obj_prop = str(props.get("objective", "") or "")
        cons_prop = str(props.get("constraints", "") or "")
        bounds_prop = str(props.get("bounds", "") or "")
        integ_prop = str(props.get("integrality", "") or "")
        sense = str(props.get("sense", "min")).lower()

        var_names = [sanitize_name(v.strip()) for v in re.split(r"[;,\\s]+", vars_prop) if v.strip()]
        if not var_names:
            seen = []
            for v in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f"{obj_prop}\n{cons_prop}"):
                v_norm = sanitize_name(v)
                if v_norm and v_norm not in seen:
                    seen.append(v_norm)
            var_names = seen

        def parse_linear_expr(expr: str, vars_list: List[str]) -> List[float]:
            coeffs = [0.0] * len(vars_list)
            expr_clean = expr.replace("*", "")
            expr_clean = expr_clean.replace("-", "+-")
            for term in expr_clean.split("+"):
                term = term.strip()
                if not term:
                    continue
                m = re.match(r"([+-]?\d*\.?\d*)?([A-Za-z_]\w*)", term)
                if not m:
                    continue
                coeff_str, var = m.groups()
                if var not in vars_list:
                    continue
                if coeff_str in ("", "+", None):
                    coeff = 1.0
                elif coeff_str == "-":
                    coeff = -1.0
                else:
                    try:
                        coeff = float(coeff_str)
                    except Exception:
                        coeff = 0.0
                coeffs[vars_list.index(var)] += coeff
            return coeffs

        c_vec = parse_linear_expr(obj_prop, var_names) if var_names else []
        if sense == "max":
            c_vec = [-v for v in c_vec]

        A_ub: List[List[float]] = []
        b_ub: List[float] = []
        A_eq: List[List[float]] = []
        b_eq: List[float] = []
        ub_labels: List[str] = []
        eq_labels: List[str] = []

        for line in re.split(r"[;\\n]", cons_prop):
            ln = line.strip()
            if not ln or not var_names:
                continue
            label = None
            if ":" in ln:
                label_part, ln = ln.split(":", 1)
                if label_part.strip():
                    label = sanitize_name(label_part.strip())
                ln = ln.strip()
            m = re.match(r"(.+?)(<=|>=|=)(.+)")
            if not m:
                continue
            lhs, op, rhs = m.groups()
            coeffs = parse_linear_expr(lhs, var_names)
            try:
                rhs_val = float(rhs.strip())
            except Exception:
                continue
            if op == ">=":
                coeffs = [-c for c in coeffs]
                rhs_val = -rhs_val
                A_ub.append(coeffs)
                b_ub.append(rhs_val)
                ub_labels.append(label or f"ub_{len(ub_labels)}")
            elif op == "<=":
                A_ub.append(coeffs)
                b_ub.append(rhs_val)
                ub_labels.append(label or f"ub_{len(ub_labels)}")
            else:
                A_eq.append(coeffs)
                b_eq.append(rhs_val)
                eq_labels.append(label or f"eq_{len(eq_labels)}")

        # Bounds parsing: "0,10;0,inf" or "x1:0,10"
        bounds_tokens = []
        for line in re.split(r"[;\\n]", bounds_prop):
            ln = line.strip()
            if not ln:
                continue
            vname = None
            if ":" in ln:
                vpart, ln = ln.split(":", 1)
                vname = sanitize_name(vpart.strip())
            parts = [p.strip() for p in ln.split(",") if p.strip()]
            lb = None
            ub = None
            if parts:
                try:
                    lb = float(parts[0]) if parts[0].lower() not in ["none", "inf", "+inf"] else None
                except Exception:
                    lb = None
            if len(parts) > 1:
                try:
                    ub = float(parts[1]) if parts[1].lower() not in ["none", "inf", "+inf"] else None
                except Exception:
                    ub = None
            bounds_tokens.append((vname, lb, ub))

        bounds_final: List[Tuple[float, float]] = []
        for idx, v in enumerate(var_names):
            match = next((b for b in bounds_tokens if b[0] == v), None)
            if match:
                bounds_final.append((match[1], match[2]))
            elif idx < len(bounds_tokens):
                bounds_final.append((bounds_tokens[idx][1], bounds_tokens[idx][2]))
            else:
                bounds_final.append((None, None))

        # Integrality parsing: "x1:int;x2:bin" or "0,1,2"
        def parse_integ_token(tok: str) -> int:
            t = tok.strip().lower()
            if t in ["2", "bin", "binary", "bool"]:
                return 2
            if t in ["1", "int", "integer"]:
                return 1
            return 0

        integ_map = {}
        integ_list: List[int] = []
        for tok in re.split(r"[;\\n,]", integ_prop):
            if not tok.strip():
                continue
            if ":" in tok:
                name_part, val_part = tok.split(":", 1)
                integ_map[sanitize_name(name_part.strip())] = parse_integ_token(val_part)
            else:
                integ_list.append(parse_integ_token(tok))

        integ_final: List[int] = []
        for idx, v in enumerate(var_names):
            if v in integ_map:
                integ_final.append(integ_map[v])
            elif idx < len(integ_list):
                integ_final.append(integ_list[idx])
            else:
                integ_final.append(0)

        name = default_var_name(node)
        c_var = f"{name}_c"
        aub_var = f"{name}_A_ub"
        bub_var = f"{name}_b_ub"
        aeq_var = f"{name}_A_eq"
        beq_var = f"{name}_b_eq"
        bounds_var = f"{name}_bounds"
        integ_var = f"{name}_integrality"
        labels_var = f"{name}_labels"
        ub_labels_var = f"{name}_labels_ub"
        eq_labels_var = f"{name}_labels_eq"
        add(f"# Linear model builder -> matrices/labels from text")
        add(f"{c_var} = np.array({c_vec!r}, dtype=float) if {bool(c_vec)} else np.array([])")
        add(f"{aub_var} = np.array({A_ub!r}, dtype=float) if {bool(A_ub)} else None")
        add(f"{bub_var} = np.array({b_ub!r}, dtype=float) if {bool(b_ub)} else None")
        add(f"{aeq_var} = np.array({A_eq!r}, dtype=float) if {bool(A_eq)} else None")
        add(f"{beq_var} = np.array({b_eq!r}, dtype=float) if {bool(b_eq)} else None")
        add(f"{bounds_var} = {bounds_final!r}")
        add(f"{integ_var} = np.array({integ_final!r}, dtype=int) if {bool(integ_final)} else None")
        add(f"{labels_var} = {var_names!r}")
        add(f"{ub_labels_var} = {ub_labels!r}")
        add(f"{eq_labels_var} = {eq_labels!r}")
        return {
            0: c_var,
            1: aub_var,
            2: bub_var,
            3: aeq_var,
            4: beq_var,
            5: bounds_var,
            6: integ_var,
            7: ub_labels_var,
            8: eq_labels_var,
            9: labels_var,
        }
    def gen_variable_builder(node, inputs):
        raw = str(node.get("properties", {}).get("variables", ""))
        name = default_var_name(node)
        import re
        c = []
        bounds = []
        integrality = []
        labels = []
        idx = 0
        for line in re.split(r"[;\n]", raw):
            ln = line.strip()
            if not ln:
                continue
            label = None
            if ":" in ln:
                label_part, ln = ln.split(":", 1)
                if label_part.strip():
                    label = sanitize_name(label_part.strip())
                ln = ln.strip()
            parts = [p.strip() for p in ln.split(",") if p.strip()]
            if not parts:
                continue
            try:
                cost = float(parts[0])
            except Exception:
                cost = 0.0
            def _get_bound(val):
                if not val:
                    return None
                val_l = val.lower()
                if val_l in ["none", "inf", "+inf"]:
                    return None
                try:
                    return float(val)
                except Exception:
                    return None
            lb = _get_bound(parts[1]) if len(parts) > 1 else None
            ub = _get_bound(parts[2]) if len(parts) > 2 else None
            try:
                integ = int(float(parts[3])) if len(parts) > 3 else 0
            except Exception:
                integ = 0
            c.append(cost)
            bounds.append((lb, ub))
            integrality.append(integ)
            labels.append(label or f"x{idx}")
            idx += 1
        c_var = f"{name}_c"
        bounds_var = f"{name}_bounds"
        integ_var = f"{name}_integrality"
        labels_var = f"{name}_labels"
        add(f"# Variable builder -> costs, bounds, integrality, labels")
        add(f"{c_var} = np.array({c!r}, dtype=float)")
        add(f"{bounds_var} = {bounds!r}")
        add(f"{integ_var} = np.array({integrality!r}, dtype=int)")
        add(f"{labels_var} = {labels!r}")
        return {0: c_var, 1: bounds_var, 2: integ_var, 3: labels_var}

    def gen_solution_report(node, inputs):
        x = inputs[0][1] if len(inputs) > 0 else "None"
        obj = inputs[1][1] if len(inputs) > 1 else "None"
        status = inputs[2][1] if len(inputs) > 2 else "''"
        dual_ub = inputs[3][1] if len(inputs) > 3 else "None"
        dual_eq = inputs[4][1] if len(inputs) > 4 else "None"
        reduced = inputs[5][1] if len(inputs) > 5 else "None"
        name = default_var_name(node)
        add(f"# Pack optimization results for display")
        add(f"{name} = {{'status': {status}, 'objective': {obj}, 'solution': {x}, 'dual_ub': {dual_ub}, 'dual_eq': {dual_eq}, 'reduced_cost': {reduced}}}")
        add(f"{name}_text = f\"status: {status}\nobjective: {obj}\nsolution: {x}\"")
        return {0: name, 1: f"{name}_text"}

    def gen_multiobjective_weighted(node, inputs):
        objs = inputs[0][1] if len(inputs) > 0 else "np.zeros((1,1))"
        weights = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        add(f"# Multi-objective weighted sum -> single objective vector")
        add(f"_objs_{name} = np.atleast_2d(np.asarray({objs}, dtype=float))")
        add(f"_m_{name}, _n_{name} = _objs_{name}.shape")
        add(f"_w_{name} = np.asarray({weights}, dtype=float).ravel() if {weights} is not None else np.ones(_m_{name})")
        add(f"if _w_{name}.size != _m_{name}: _w_{name} = np.ones(_m_{name})")
        add(f"_w_{name} = _w_{name} / (_w_{name}.sum() if _w_{name}.sum() != 0 else 1)")
        add(f"{name} = _w_{name} @ _objs_{name}")
        return name

    def gen_export_lp(node, inputs):
        c = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        A_ub = inputs[1][1] if len(inputs) > 1 else "None"
        b_ub = inputs[2][1] if len(inputs) > 2 else "None"
        A_eq = inputs[3][1] if len(inputs) > 3 else "None"
        b_eq = inputs[4][1] if len(inputs) > 4 else "None"
        bounds_in = inputs[5][1] if len(inputs) > 5 else None
        integrality_in = inputs[6][1] if len(inputs) > 6 else None
        labels_in = inputs[7][1] if len(inputs) > 7 else None
        name_prop = node.get("properties", {}).get("name", "model")
        name = default_var_name(node)
        add(f"# Export LP-like text (copyable to solvers)")
        add(f"_c_lp_{name} = np.asarray({c}, dtype=float).ravel()")
        add(f"_Aub_lp_{name} = np.asarray({A_ub}, dtype=float) if {A_ub} is not None else None")
        add(f"_bub_lp_{name} = np.asarray({b_ub}, dtype=float).ravel() if {b_ub} is not None else None")
        add(f"_Aeq_lp_{name} = np.asarray({A_eq}, dtype=float) if {A_eq} is not None else None")
        add(f"_beq_lp_{name} = np.asarray({b_eq}, dtype=float).ravel() if {b_eq} is not None else None")
        add(f"_bounds_lp_{name} = {bounds_in} if {bounds_in} is not None else None")
        add(f"if isinstance(_bounds_lp_{name}, np.ndarray): _bounds_lp_{name} = _bounds_lp_{name}.tolist()")
        add(f"_integ_lp_{name} = np.asarray({integrality_in}, dtype=int) if {integrality_in} is not None else None")
        add(f"_labels_lp_{name} = list({labels_in}) if {labels_in} is not None else [f'x{{i}}' for i in range(len(_c_lp_{name}))]")
        add(f"if len(_labels_lp_{name}) != len(_c_lp_{name}): _labels_lp_{name} = [f'x{{i}}' for i in range(len(_c_lp_{name}))]")
        add(f"_lines_lp_{name} = []")
        add(f"_lines_lp_{name}.append('\\ Problem: {name_prop}')")
        add(f"_obj_terms_{name} = []")
        add(f"for _i, _coef in enumerate(_c_lp_{name}):")
        add(f"    if _coef != 0: _obj_terms_{name}.append(f'{{_coef:g}} {{_labels_lp_{name}[_i]}}')")
        add(f"_lines_lp_{name}.append('Minimize: ' + (' + '.join(_obj_terms_{name}) if _obj_terms_{name} else '0'))")
        add(f"_lines_lp_{name}.append('Subject To:')")
        add(f"if _Aub_lp_{name} is not None and _bub_lp_{name} is not None:")
        add(f"    for _row, _rhs in zip(_Aub_lp_{name}, _bub_lp_{name}):")
        add(f"        _terms = [f'{{_row[i]:g}} {{_labels_lp_{name}[i]}}' for i in range(len(_row)) if _row[i] != 0]")
        add(f"        _lines_lp_{name}.append('  ' + (' + '.join(_terms) if _terms else '0') + f' <= {{_rhs:g}}')")
        add(f"if _Aeq_lp_{name} is not None and _beq_lp_{name} is not None:")
        add(f"    for _row, _rhs in zip(_Aeq_lp_{name}, _beq_lp_{name}):")
        add(f"        _terms = [f'{{_row[i]:g}} {{_labels_lp_{name}[i]}}' for i in range(len(_row)) if _row[i] != 0]")
        add(f"        _lines_lp_{name}.append('  ' + (' + '.join(_terms) if _terms else '0') + f' = {{_rhs:g}}')")
        add(f"_lines_lp_{name}.append('Bounds:')")
        add(f"if _bounds_lp_{name}:")
        add(f"    for _i, (_lb, _ub) in enumerate(_bounds_lp_{name}):")
        add(f"        _lb_txt = '-inf' if _lb in [None, -np.inf] else f'{{_lb:g}}'")
        add(f"        _ub_txt = 'inf' if _ub in [None, np.inf] else f'{{_ub:g}}'")
        add(f"        _lines_lp_{name}.append(f'  {{_lb_txt}} <= {{_labels_lp_{name}[_i]}} <= {{_ub_txt}}')")
        add(f"_lines_lp_{name}.append('Integrality:')")
        add(f"if _integ_lp_{name} is not None and _integ_lp_{name}.size:")
        add(f"    for _i, _itype in enumerate(_integ_lp_{name}):")
        add(f"        if _itype == 1: _lines_lp_{name}.append(f'  int {{_labels_lp_{name}[_i]}}')")
        add(f"        if _itype == 2: _lines_lp_{name}.append(f'  bin {{_labels_lp_{name}[_i]}}')")
        add(f"{name} = '\n'.join(_lines_lp_{name})")
        return name
    def gen_quadratic_programming(node, inputs):
        Q = inputs[0][1] if len(inputs) > 0 else "np.eye(2)"
        c = inputs[1][1] if len(inputs) > 1 else "np.zeros(2)"
        name = default_var_name(node)
        add(f"# Quadratic Programming (Minimize 0.5 x^T Q x + c^T x)")
        add(f"def obj_{name}(x): return 0.5 * x @ {Q} @ x + {c} @ x")
        add(f"res_{name} = scipy.optimize.minimize(obj_{name}, np.zeros({c}.shape))")
        add(f"{name} = res_{name}.x")
        return name

    def gen_dynamic_programming(node, inputs):
        values = inputs[0][1] if len(inputs) > 0 else "np.array([60, 100, 120])"
        weights = inputs[1][1] if len(inputs) > 1 else "np.array([10, 20, 30])"
        name = default_var_name(node)
        capacity = node.get("properties", {}).get("capacity", 50)
        problem_type = node.get("properties", {}).get("problem_type", "knapsack")
        add(f"# Dynamic Programming - {problem_type}")
        add(f"def dp_knapsack_{name}(values, weights, capacity):")
        add(f"    '''0/1背包问题的动态规划解法'''")
        add(f"    n = len(values)")
        add(f"    dp = np.zeros((n + 1, capacity + 1))")
        add(f"    for i in range(1, n + 1):")
        add(f"        for w in range(capacity + 1):")
        add(f"            if weights[i-1] <= w:")
        add(f"                dp[i][w] = max(dp[i-1][w], dp[i-1][w-int(weights[i-1])] + values[i-1])")
        add(f"            else:")
        add(f"                dp[i][w] = dp[i-1][w]")
        add(f"    # 回溯找出选中的物品")
        add(f"    selected = []")
        add(f"    w = capacity")
        add(f"    for i in range(n, 0, -1):")
        add(f"        if dp[i][w] != dp[i-1][w]:")
        add(f"            selected.append(i-1)")
        add(f"            w -= int(weights[i-1])")
        add(f"    return dp[n][capacity], np.array(selected[::-1])")
        add(f"")
        add(f"_values_{name} = np.array({values}) if hasattr({values}, '__len__') else np.array([60, 100, 120])")
        add(f"_weights_{name} = np.array({weights}) if hasattr({weights}, '__len__') else np.array([10, 20, 30])")
        add(f"{name}_max, {name}_selected = dp_knapsack_{name}(_values_{name}, _weights_{name}, {capacity})")
        add(f"{name} = {name}_max  # 最大价值")
        return name

    def gen_backtracking(node, inputs):
        constraints = inputs[0][1] if len(inputs) > 0 else "None"
        name = default_var_name(node)
        n = node.get("properties", {}).get("n", 8)
        problem_type = node.get("properties", {}).get("problem_type", "n_queens")
        add(f"# Backtracking Search - {problem_type}")
        add(f"def backtrack_nqueens_{name}(n):")
        add(f"    '''N皇后问题: 返回所有解'''")
        add(f"    solutions = []")
        add(f"    def solve(row, cols, diag1, diag2, path):")
        add(f"        if row == n:")
        add(f"            solutions.append(path[:])")
        add(f"            return")
        add(f"        for c in range(n):")
        add(f"            if c not in cols and (row-c) not in diag1 and (row+c) not in diag2:")
        add(f"                path.append(c)")
        add(f"                solve(row+1, cols|{{c}}, diag1|{{row-c}}, diag2|{{row+c}}, path)")
        add(f"                path.pop()")
        add(f"    solve(0, set(), set(), set(), [])")
        add(f"    return solutions")
        add(f"")
        add(f"def backtrack_subset_sum_{name}(nums, target):")
        add(f"    '''子集和问题: 找出和为target的子集'''")
        add(f"    result = []")
        add(f"    def backtrack(start, path, current_sum):")
        add(f"        if current_sum == target:")
        add(f"            result.append(path[:])")
        add(f"            return")
        add(f"        if current_sum > target:")
        add(f"            return")
        add(f"        for i in range(start, len(nums)):")
        add(f"            path.append(nums[i])")
        add(f"            backtrack(i + 1, path, current_sum + nums[i])")
        add(f"            path.pop()")
        add(f"    backtrack(0, [], 0)")
        add(f"    return result")
        add(f"")
        add(f"def backtrack_permutations_{name}(arr):")
        add(f"    '''全排列问题'''")
        add(f"    result = []")
        add(f"    def backtrack(path, remaining):")
        add(f"        if not remaining:")
        add(f"            result.append(path[:])")
        add(f"            return")
        add(f"        for i in range(len(remaining)):")
        add(f"            path.append(remaining[i])")
        add(f"            backtrack(path, remaining[:i] + remaining[i+1:])")
        add(f"            path.pop()")
        add(f"    backtrack([], list(arr))")
        add(f"    return result")
        add(f"")
        add(f"# 根据问题类型选择算法")
        add(f"if '{problem_type}' == 'n_queens':")
        add(f"    _solutions_{name} = backtrack_nqueens_{name}({n})")
        add(f"    {name} = np.array(_solutions_{name}[0]) if _solutions_{name} else np.array([])")
        add(f"elif '{problem_type}' == 'subset_sum' and {constraints} is not None:")
        add(f"    _solutions_{name} = backtrack_subset_sum_{name}(list({constraints}.flatten()), {n})")
        add(f"    {name} = np.array(_solutions_{name}[0]) if _solutions_{name} else np.array([])")
        add(f"elif '{problem_type}' == 'permutation' and {constraints} is not None:")
        add(f"    _solutions_{name} = backtrack_permutations_{name}(list({constraints}.flatten()))")
        add(f"    {name} = np.array(_solutions_{name}[0]) if _solutions_{name} else np.array([])")
        add(f"else:")
        add(f"    _solutions_{name} = backtrack_nqueens_{name}({n})")
        add(f"    {name} = np.array(_solutions_{name}[0]) if _solutions_{name} else np.array([])")
        return name

    def gen_divide_conquer(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        name = default_var_name(node)
        operation = node.get("properties", {}).get("operation", "sort")
        add(f"# Divide & Conquer - {operation}")
        add(f"def merge_sort_{name}(arr):")
        add(f"    '''归并排序'''")
        add(f"    if len(arr) <= 1:")
        add(f"        return arr")
        add(f"    mid = len(arr) // 2")
        add(f"    left = merge_sort_{name}(arr[:mid])")
        add(f"    right = merge_sort_{name}(arr[mid:])")
        add(f"    return np.array(sorted(list(left) + list(right)))")
        add(f"")
        add(f"def quick_select_{name}(arr, k):")
        add(f"    '''快速选择: 找第k小的元素'''")
        add(f"    if len(arr) == 1:")
        add(f"        return arr[0]")
        add(f"    pivot = arr[len(arr) // 2]")
        add(f"    left = [x for x in arr if x < pivot]")
        add(f"    middle = [x for x in arr if x == pivot]")
        add(f"    right = [x for x in arr if x > pivot]")
        add(f"    if k < len(left):")
        add(f"        return quick_select_{name}(np.array(left), k)")
        add(f"    elif k < len(left) + len(middle):")
        add(f"        return pivot")
        add(f"    else:")
        add(f"        return quick_select_{name}(np.array(right), k - len(left) - len(middle))")
        add(f"")
        add(f"def max_subarray_{name}(arr):")
        add(f"    '''最大子数组和 (分治)'''")
        add(f"    def helper(arr, low, high):")
        add(f"        if low == high:")
        add(f"            return arr[low]")
        add(f"        mid = (low + high) // 2")
        add(f"        left_max = helper(arr, low, mid)")
        add(f"        right_max = helper(arr, mid + 1, high)")
        add(f"        # 跨越中点的最大子数组")
        add(f"        left_sum = float('-inf')")
        add(f"        temp_sum = 0")
        add(f"        for i in range(mid, low - 1, -1):")
        add(f"            temp_sum += arr[i]")
        add(f"            left_sum = max(left_sum, temp_sum)")
        add(f"        right_sum = float('-inf')")
        add(f"        temp_sum = 0")
        add(f"        for i in range(mid + 1, high + 1):")
        add(f"            temp_sum += arr[i]")
        add(f"            right_sum = max(right_sum, temp_sum)")
        add(f"        cross_max = left_sum + right_sum")
        add(f"        return max(left_max, right_max, cross_max)")
        add(f"    return helper(arr, 0, len(arr) - 1) if len(arr) > 0 else 0")
        add(f"")
        add(f"_data_{name} = np.array({data}) if hasattr({data}, '__len__') else np.array([])")
        add(f"if '{operation}' == 'sort':")
        add(f"    {name} = merge_sort_{name}(_data_{name}) if len(_data_{name}) > 0 else np.array([])")
        add(f"elif '{operation}' == 'median':")
        add(f"    {name} = quick_select_{name}(_data_{name}, len(_data_{name}) // 2) if len(_data_{name}) > 0 else 0")
        add(f"elif '{operation}' == 'max_subarray':")
        add(f"    {name} = max_subarray_{name}(_data_{name}) if len(_data_{name}) > 0 else 0")
        add(f"else:")
        add(f"    {name} = merge_sort_{name}(_data_{name}) if len(_data_{name}) > 0 else np.array([])")
        return name

    def gen_neural_network_opt(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        hidden = node.get("properties", {}).get("hidden", "10,10")
        hidden_tuple = tuple(int(x) for x in hidden.split(',') if x.strip())
        add(f"# Neural Network Optimization")
        add(f"{name} = MLPRegressor(hidden_layer_sizes={hidden_tuple}, max_iter=200)")
        add(f"if {X} is not None and {y} is not None: {name}.fit({X}, {y})")
        return name

    def gen_grid_search(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        param_grid = node.get("properties", {}).get("param_grid", "C:0.1,1,10")
        add(f"# Grid Search (SVM)")
        add(f"from sklearn.model_selection import GridSearchCV")
        add(f"_param_grid_{name} = dict()")
        add(f"for part in '{param_grid}'.split(';'):")
        add(f"    k, v = part.split(':')")
        add(f"    _param_grid_{name}[k.strip()] = [float(x) for x in v.split(',')]")
        add(f"_clf_{name} = GridSearchCV(SVC(), _param_grid_{name}, cv=3)")
        add(f"if {data} is not None and {y} is not None: _clf_{name}.fit({data}, {y})")
        add(f"{name} = np.array(list(_clf_{name}.best_params_.values())) if {data} is not None else np.array([])")
        return name

    def gen_exhaustive_search(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        name = default_var_name(node)
        add(f"# Exhaustive Search (Find max value)")
        add(f"{name} = np.max({data}) if {data}.size else 0")
        return name

    def gen_discretize(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        name = default_var_name(node)
        bins = int(node.get("properties", {}).get("bins", 10))
        add(f"# Discretize")
        add(f"try:")
        add(f"    {name} = np.digitize({data}, np.linspace({data}.min(), {data}.max(), {bins}))")
        add(f"except Exception: {name} = []")
        return name

    def gen_root_finding(node, inputs):
        coeffs = inputs[0][1] if len(inputs) > 0 else "np.array([1, 0, -4])"
        name = default_var_name(node)
        add(f"# Root Finding (Polynomial roots)")
        add(f"{name} = np.roots({coeffs}) if {coeffs}.size else np.array([])")
        return name

    def gen_image_filter(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.zeros((5,5))"
        name = default_var_name(node)
        ftype = node.get("properties", {}).get("type", "blur")
        kernel = node.get("properties", {}).get("kernel", 3)
        add(f"# Image Filter ({ftype})")
        add(f"from scipy.ndimage import gaussian_filter, uniform_filter")
        if ftype == "gaussian":
            add(f"{name} = gaussian_filter({data}, sigma={kernel})")
        else:
            add(f"{name} = uniform_filter({data}, size={kernel})")
        return name

    # --- 2. Models ---

    def gen_bp_nn(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        hidden = node.get("properties", {}).get("hidden_layers", "10,10")
        max_iter = node.get("properties", {}).get("max_iter", 500)
        hidden_tuple = tuple(int(x) for x in hidden.split(',') if x.strip())
        add(f"# BP Neural Network")
        add(f"{name} = MLPRegressor(hidden_layer_sizes={hidden_tuple}, max_iter={max_iter})")
        add(f"if {X} is not None and {y} is not None: {name}.fit({X}, {y})")
        return name

    def gen_poly_fit(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        y = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        deg = int(node.get("properties", {}).get("degree", 2))
        add(f"# Polynomial Fitting")
        add(f"{name} = np.polyfit({X}, {y}, {deg}) if {X}.size and {y}.size else np.array([])")
        return name

    def gen_svm(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        kernel = node.get("properties", {}).get("kernel", "rbf")
        C = node.get("properties", {}).get("C", 1.0)
        add(f"# SVM Prediction (SVR)")
        add(f"{name} = SVR(kernel='{kernel}', C={C})")
        add(f"if {X} is not None and {y} is not None: {name}.fit({X}, {y})")
        return name

    def gen_grey_prediction(node, inputs):
        series = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        name = default_var_name(node)
        steps = node.get("properties", {}).get("steps", 5)
        add(f"# Grey Prediction GM(1,1)")
        add(f"def gm11_{name}(x0, pred_steps):")
        add(f"    x1 = np.cumsum(x0)")
        add(f"    z1 = (x1[:-1] + x1[1:]) / 2.0")
        add(f"    B = np.append(-z1.reshape(-1,1), np.ones((len(z1),1)), axis=1)")
        add(f"    Y = x0[1:].reshape(-1,1)")
        add(f"    [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)")
        add(f"    def f(k): return (x0[0]-b/a)*np.exp(-a*k) + b/a")
        add(f"    return np.array([f(k) for k in range(len(x0)+pred_steps)])")
        add(f"{name} = gm11_{name}({series}, {steps}) if {series}.size > 3 else np.array([])")
        return name

    def gen_time_series(node, inputs):
        series = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        name = default_var_name(node)
        p = node.get("properties", {}).get("p", 1)
        d = node.get("properties", {}).get("d", 1)
        q = node.get("properties", {}).get("q", 1)
        steps = node.get("properties", {}).get("steps", 5)
        add(f"# Time Series (ARIMA)")
        add(f"try:")
        add(f"    model_{name} = sm.tsa.ARIMA({series}, order=({p},{d},{q})).fit()")
        add(f"    {name} = model_{name}.forecast(steps={steps})")
        add(f"except Exception: {name} = np.array([])")
        return name

    def gen_markov_chain(node, inputs):
        matrix = inputs[0][1] if len(inputs) > 0 else "np.eye(3)"
        state = inputs[1][1] if len(inputs) > 1 else "np.array([1,0,0])"
        name = default_var_name(node)
        steps = node.get("properties", {}).get("steps", 1)
        add(f"# Markov Chain Prediction")
        add(f"_state_{name} = {state}")
        add(f"for _ in range({steps}): _state_{name} = _state_{name} @ {matrix}")
        add(f"{name} = _state_{name}")
        return name

    # --- 3. Eval ---

    def gen_ahp(node, inputs):
        criteria = inputs[0][1] if len(inputs) > 0 else "np.eye(3)"
        name = default_var_name(node)
        add(f"# AHP (Analytic Hierarchy Process)")
        add(f"eig_val_{name}, eig_vec_{name} = np.linalg.eig({criteria})")
        add(f"max_eig_ind_{name} = np.argmax(eig_val_{name})")
        add(f"{name} = np.real(eig_vec_{name}[:, max_eig_ind_{name}])")
        add(f"{name} = {name} / {name}.sum()")
        return name

    def gen_fuzzy_eval(node, inputs):
        R = inputs[0][1] if len(inputs) > 0 else "np.eye(3)"
        W = inputs[1][1] if len(inputs) > 1 else "np.ones(3)/3"
        name = default_var_name(node)
        add(f"# Fuzzy Comprehensive Evaluation")
        add(f"{name} = {W} @ {R}")
        return name

    def gen_grey_relational(node, inputs):
        ref = inputs[0][1] if len(inputs) > 0 else "np.zeros(5)"
        comp = inputs[1][1] if len(inputs) > 1 else "np.zeros((3,5))"
        name = default_var_name(node)
        rho = node.get("properties", {}).get("rho", 0.5)
        add(f"# Grey Relational Analysis")
        add(f"rho_{name} = {rho}")
        add(f"diff_{name} = np.abs({comp} - {ref})")
        add(f"min_diff_{name} = diff_{name}.min()")
        add(f"max_diff_{name} = diff_{name}.max()")
        add(f"coef_{name} = (min_diff_{name} + rho_{name} * max_diff_{name}) / (diff_{name} + rho_{name} * max_diff_{name})")
        add(f"{name} = coef_{name}.mean(axis=1)")
        return name

    def gen_rsr(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.zeros((5,3))"
        name = default_var_name(node)
        add(f"# Rank Sum Ratio (RSR)")
        add(f"ranks_{name} = scipy.stats.rankdata({data}, axis=0)")
        add(f"{name} = ranks_{name}.mean(axis=1) / {data}.shape[0]")
        return name

    def gen_coupling(node, inputs):
        sysA = inputs[0][1] if len(inputs) > 0 else "0.5"
        sysB = inputs[1][1] if len(inputs) > 1 else "0.5"
        name = default_var_name(node)
        alpha = node.get("properties", {}).get("alpha", 0.5)
        add(f"# Coupling Coordination Degree")
        add(f"# Compute mean if array")
        add(f"_a_{name} = np.mean({sysA}) if hasattr({sysA}, '__len__') else {sysA}")
        add(f"_b_{name} = np.mean({sysB}) if hasattr({sysB}, '__len__') else {sysB}")
        add(f"C_{name} = 2 * np.sqrt((_a_{name} * _b_{name}) / (_a_{name} + _b_{name} + 1e-10)**2)")
        add(f"T_{name} = {alpha} * _a_{name} + (1-{alpha}) * _b_{name}")
        add(f"{name} = np.sqrt(C_{name} * T_{name})")
        return name

    def gen_bp_eval(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        hidden = node.get("properties", {}).get("hidden", "10")
        hidden_tuple = tuple(int(x) for x in hidden.split(',') if x.strip())
        add(f"# BP NN Evaluation")
        add(f"_model_{name} = MLPRegressor(hidden_layer_sizes={hidden_tuple}, max_iter=500)")
        add(f"if {X} is not None and {y} is not None:")
        add(f"    _model_{name}.fit({X}, {y})")
        add(f"    {name} = _model_{name}.predict({X})")
        add(f"else: {name} = np.array([])")
        return name

    def gen_topsis(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.zeros((3,3))"
        weights = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        benefit = node.get("properties", {}).get("benefit", "1,1,1")
        add(f"# TOPSIS")
        add(f"_benefit_{name} = np.array([int(x) for x in '{benefit}'.split(',')]) if '{benefit}' else None")
        add(f"norm_{name} = {data} / np.sqrt(({data}**2).sum(axis=0) + 1e-10)")
        add(f"w_{name} = {weights} if {weights} is not None else np.ones({data}.shape[1])/{data}.shape[1]")
        add(f"weighted_{name} = norm_{name} * w_{name}")
        add(f"# Determine ideal best/worst based on benefit/cost criteria")
        add(f"if _benefit_{name} is not None and len(_benefit_{name}) == {data}.shape[1]:")
        add(f"    best_{name} = np.where(_benefit_{name}==1, weighted_{name}.max(axis=0), weighted_{name}.min(axis=0))")
        add(f"    worst_{name} = np.where(_benefit_{name}==1, weighted_{name}.min(axis=0), weighted_{name}.max(axis=0))")
        add(f"else:")
        add(f"    best_{name} = weighted_{name}.max(axis=0)")
        add(f"    worst_{name} = weighted_{name}.min(axis=0)")
        add(f"d_best_{name} = np.sqrt(((weighted_{name} - best_{name})**2).sum(axis=1))")
        add(f"d_worst_{name} = np.sqrt(((weighted_{name} - worst_{name})**2).sum(axis=1))")
        add(f"{name} = d_worst_{name} / (d_best_{name} + d_worst_{name} + 1e-10)")
        return name

    def gen_pca(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.zeros((5,2))"
        name = default_var_name(node)
        n_comp = int(node.get("properties", {}).get("n_components", 2))
        add(f"# PCA")
        add(f"{name} = PCA(n_components={n_comp}).fit_transform({data}) if {data}.size else np.array([])")
        return name

    # --- 4. Classification ---

    def gen_kmeans(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.zeros((5,2))"
        name = default_var_name(node)
        k = int(node.get("properties", {}).get("k", 3))
        add(f"# K-Means")
        add(f"{name} = KMeans(n_clusters={k}).fit_predict({data}) if {data}.size else np.array([])")
        return name

    def gen_decision_tree(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        max_depth = node.get("properties", {}).get("max_depth", 5)
        add(f"# Decision Tree")
        add(f"{name} = DecisionTreeClassifier(max_depth={max_depth})")
        add(f"if {X} is not None and {y} is not None: {name}.fit({X}, {y})")
        return name

    def gen_logistic_regression(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        C = node.get("properties", {}).get("C", 1.0)
        add(f"# Logistic Regression")
        add(f"{name} = LogisticRegression(C={C})")
        add(f"if {X} is not None and {y} is not None: {name}.fit({X}, {y})")
        return name

    def gen_random_forest(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        n_est = node.get("properties", {}).get("n_estimators", 100)
        max_depth = node.get("properties", {}).get("max_depth", 5)
        add(f"# Random Forest")
        add(f"{name} = RandomForestClassifier(n_estimators={n_est}, max_depth={max_depth})")
        add(f"if {X} is not None and {y} is not None: {name}.fit({X}, {y})")
        return name

    def gen_naive_bayes(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        add(f"# Naive Bayes")
        add(f"{name} = GaussianNB()")
        add(f"if {X} is not None and {y} is not None: {name}.fit({X}, {y})")
        return name

    # --- 5. Optimization ---

    def gen_knapsack(node, inputs):
        weights = inputs[0][1] if len(inputs) > 0 else "[]"
        values = inputs[1][1] if len(inputs) > 1 else "[]"
        name = default_var_name(node)
        cap = node.get("properties", {}).get("capacity", 50)
        add(f"# Knapsack (0/1) DP")
        add(f"def solve_knapsack_{name}(w, v, c):")
        add(f"    w, v = list(w), list(v)")
        add(f"    n = len(w)")
        add(f"    if n == 0: return [], 0")
        add(f"    dp = [[0]*(int(c)+1) for _ in range(n+1)]")
        add(f"    for i in range(1, n+1):")
        add(f"        for j in range(int(c)+1):")
        add(f"            dp[i][j] = dp[i-1][j]")
        add(f"            if int(w[i-1]) <= j:")
        add(f"                dp[i][j] = max(dp[i][j], dp[i-1][j-int(w[i-1])] + v[i-1])")
        add(f"    # Backtrack to find selected items")
        add(f"    selected = []")
        add(f"    j = int(c)")
        add(f"    for i in range(n, 0, -1):")
        add(f"        if dp[i][j] != dp[i-1][j]:")
        add(f"            selected.append(i-1)")
        add(f"            j -= int(w[i-1])")
        add(f"    return selected[::-1], dp[n][int(c)]")
        add(f"{name}_items, {name}_value = solve_knapsack_{name}({weights}, {values}, {cap})")
        add(f"{name} = np.array({name}_items)")
        return name

    def gen_tsp(node, inputs):
        dist = inputs[0][1] if len(inputs) > 0 else "np.zeros((0,0))"
        name = default_var_name(node)
        add(f"# TSP (Greedy Approx)")
        add(f"G_{name} = nx.from_numpy_array({dist})")
        add(f"try:")
        add(f"    {name}_path = nx.approximation.traveling_salesman_problem(G_{name}, cycle=True)")
        add(f"    {name}_dist = sum({dist}[{name}_path[i]][{name}_path[i+1]] for i in range(len({name}_path)-1))")
        add(f"    {name} = np.array({name}_path)")
        add(f"except Exception:")
        add(f"    {name} = np.array([])")
        add(f"    {name}_dist = 0")
        return name

    def gen_vrp(node, inputs):
        dist = inputs[0][1] if len(inputs) > 0 else "np.zeros((0,0))"
        demands = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        capacity = node.get("properties", {}).get("capacity", 100)
        add(f"# Vehicle Routing Problem (Simple Greedy)")
        add(f"def solve_vrp_{name}(dist, demands, cap):")
        add(f"    n = len(demands)")
        add(f"    if n == 0: return []")
        add(f"    visited = [False] * n")
        add(f"    visited[0] = True  # depot")
        add(f"    routes = []")
        add(f"    while not all(visited):")
        add(f"        route = [0]")
        add(f"        load = 0")
        add(f"        while True:")
        add(f"            last = route[-1]")
        add(f"            best_next, best_dist = -1, float('inf')")
        add(f"            for j in range(n):")
        add(f"                if not visited[j] and load + demands[j] <= cap and dist[last][j] < best_dist:")
        add(f"                    best_next, best_dist = j, dist[last][j]")
        add(f"            if best_next == -1: break")
        add(f"            route.append(best_next)")
        add(f"            visited[best_next] = True")
        add(f"            load += demands[best_next]")
        add(f"        route.append(0)")
        add(f"        routes.append(route)")
        add(f"    return routes")
        add(f"{name} = solve_vrp_{name}({dist}, {demands}, {capacity})")
        return name

    # --- Existing ---

    def gen_lr_fit(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else None
        y = inputs[1][1] if len(inputs) > 1 else None
        name = default_var_name(node)
        fit_intercept = bool(node.get("properties", {}).get("fit_intercept", True))
        add(f"# Linear Regression (least squares)")
        if X:
            add(f"_X = np.array({X}, dtype=float)")
        else:
            add(f"_X = np.zeros((0,0))")
        if y:
            add(f"_y = np.array({y}, dtype=float)")
        else:
            add(f"_y = np.zeros((0,))")
        if fit_intercept:
            add(f"_X_lr = np.column_stack([np.ones((_X.shape[0],)), _X]) if _X.size else _X")
        else:
            add(f"_X_lr = _X")
        add(f"_theta = np.linalg.pinv(_X_lr) @ _y if _X_lr.size and _y.size else np.array([])")
        add(f"{name} = {{'theta': _theta, 'fit_intercept': {str(fit_intercept)}}}")
        return name

    def gen_ridge(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else None
        y = inputs[1][1] if len(inputs) > 1 else None
        name = default_var_name(node)
        alpha = float(node.get("properties", {}).get("alpha", 1.0))
        fit_intercept = bool(node.get("properties", {}).get("fit_intercept", True))
        add("# Ridge Regression")
        add(f"{name} = Ridge(alpha={alpha}, fit_intercept={fit_intercept})")
        add(f"if {X} is not None and {y} is not None:")
        add(f"    {name}.fit({X}, {y})")
        return name

    def gen_lasso(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else None
        y = inputs[1][1] if len(inputs) > 1 else None
        name = default_var_name(node)
        alpha = float(node.get("properties", {}).get("alpha", 1.0))
        fit_intercept = bool(node.get("properties", {}).get("fit_intercept", True))
        add("# Lasso Regression")
        add(f"{name} = Lasso(alpha={alpha}, fit_intercept={fit_intercept})")
        add(f"if {X} is not None and {y} is not None:")
        add(f"    {name}.fit({X}, {y})")
        return name

    def gen_poly_features(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else None
        name = default_var_name(node)
        degree = int(node.get("properties", {}).get("degree", 2))
        include_bias = bool(node.get("properties", {}).get("include_bias", True))
        add("# Polynomial Features")
        add(f"_pf_{name} = PolynomialFeatures(degree={degree}, include_bias={include_bias})")
        add(f"{name} = _pf_{name}.fit_transform({X}) if {X} is not None else np.array([])")
        return name

    def gen_predict(node, inputs):
        model = inputs[0][1] if len(inputs) > 0 else None
        X = inputs[1][1] if len(inputs) > 1 else None
        name = default_var_name(node)
        add(f"# Predict using linear regression model")
        add(f"_model = {model if model else '{}'}")
        add(f"_theta = _model.get('theta', np.array([]))")
        if X:
            add(f"_X = np.array({X}, dtype=float)")
        else:
            add(f"_X = np.zeros((0,0))")
        add(f"_Xp = np.column_stack([np.ones((_X.shape[0],)), _X]) if _model.get('fit_intercept', False) and _X.size else _X")
        add(f"{name} = _Xp @ _theta if _Xp.size and _theta.size else np.array([])")
        return name

    def gen_mse(node, inputs):
        y_true = inputs[0][1] if len(inputs) > 0 else None
        y_pred = inputs[1][1] if len(inputs) > 1 else None
        name = default_var_name(node)
        add(f"{name} = float(np.mean((({y_true if y_true else '0'}) - ({y_pred if y_pred else '0'}))**2))")
        return name

    def gen_mae(node, inputs):
        y_true = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        y_pred = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add("# Mean Absolute Error")
        add(f"_y_true_{name} = np.asarray({y_true}).ravel()")
        add(f"_y_pred_{name} = np.asarray({y_pred}).ravel()")
        add(f"_len_{name} = min(_y_true_{name}.size, _y_pred_{name}.size)")
        add(f"if _len_{name}:")
        add(f"    {name} = float(np.mean(np.abs(_y_true_{name}[:_len_{name}] - _y_pred_{name}[:_len_{name}])))")
        add(f"else:")
        add(f"    {name} = 0.0")
        return name

    def gen_rmse(node, inputs):
        y_true = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        y_pred = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add("# Root Mean Square Error")
        add(f"_y_true_{name} = np.asarray({y_true}).ravel()")
        add(f"_y_pred_{name} = np.asarray({y_pred}).ravel()")
        add(f"_len_{name} = min(_y_true_{name}.size, _y_pred_{name}.size)")
        add(f"if _len_{name}:")
        add(f"    {name} = float(np.sqrt(np.mean((_y_true_{name}[:_len_{name}] - _y_pred_{name}[:_len_{name}])**2)))")
        add(f"else:")
        add(f"    {name} = 0.0")
        return name

    def gen_r2(node, inputs):
        y_true = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        y_pred = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add("# R^2 Score")
        add(f"_y_true_{name} = np.asarray({y_true}).ravel()")
        add(f"_y_pred_{name} = np.asarray({y_pred}).ravel()")
        add(f"_len_{name} = min(_y_true_{name}.size, _y_pred_{name}.size)")
        add(f"if _len_{name}:")
        add(f"    _res_{name} = _y_true_{name}[:_len_{name}] - _y_pred_{name}[:_len_{name}]")
        add(f"    _ss_res_{name} = float(np.sum(_res_{name}**2))")
        add(f"    _ss_tot_{name} = float(np.sum((_y_true_{name}[:_len_{name}] - np.mean(_y_true_{name}[:_len_{name}]))**2))")
        add(f"    {name} = float(1 - _ss_res_{name} / (_ss_tot_{name} + 1e-10))")
        add(f"else:")
        add(f"    {name} = 0.0")
        return name

    def gen_accuracy(node, inputs):
        y_true = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        y_pred = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add("# Classification Accuracy")
        add(f"_y_true_{name} = np.asarray({y_true}).ravel()")
        add(f"_y_pred_{name} = np.asarray({y_pred}).ravel()")
        add(f"_len_{name} = min(_y_true_{name}.size, _y_pred_{name}.size)")
        add(f"if _len_{name}:")
        add(f"    _acc_{name} = _y_true_{name}[:_len_{name}] == _y_pred_{name}[:_len_{name}]")
        add(f"    {name} = float(np.mean(_acc_{name}))")
        add(f"else:")
        add(f"    {name} = 0.0")
        return name

    def gen_correlation(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        Y = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add(f"# Correlation Analysis")
        add(f"{name} = np.corrcoef({X}, {Y})[0,1] if {X}.size > 1 and {Y}.size > 1 else 0")
        return name

    def gen_anova(node, inputs):
        g1 = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        g2 = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add(f"# ANOVA (Two groups)")
        add(f"try:")
        add(f"    _f_{name}, _p_{name} = scipy.stats.f_oneway({g1}, {g2})")
        add(f"    {name} = _f_{name}")
        add(f"    {name}_pvalue = _p_{name}")
        add(f"except Exception:")
        add(f"    {name} = 0")
        add(f"    {name}_pvalue = 1")
        return name

    def gen_discriminant(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        add(f"# Discriminant Analysis")
        add(f"{name} = LinearDiscriminantAnalysis()")
        add(f"if {X} is not None and {y} is not None: {name}.fit({X}, {y})")
        return name

    def gen_autocorr(node, inputs):
        series = inputs[0][1] if inputs else "np.array([])"
        name = default_var_name(node)
        nlags = int(node.get("properties", {}).get("nlags", 40))
        demean = bool(node.get("properties", {}).get("demean", True))
        add("# Auto-correlation")
        add(f"_s_{name} = np.asarray({series}).ravel()")
        add(f"if _s_{name}.size == 0:")
        add(f"    {name}_acf = np.array([])")
        add(f"else:")
        add(f"    _s_center_{name} = _s_{name} - (_s_{name}.mean() if {demean} else 0)")
        add(f"    try:")
        add(f"        {name}_acf = sm.tsa.stattools.acf(_s_center_{name}, nlags={nlags}, fft=True)")
        add(f"    except Exception:")
        add(f"        {name}_acf = np.array([])")
        add(f"{name}_lags = np.arange({name}_acf.size) if isinstance({name}_acf, np.ndarray) else np.array([])")
        return {0: f"{name}_acf", 1: f"{name}_lags"}

    def gen_pacf(node, inputs):
        series = inputs[0][1] if inputs else "np.array([])"
        name = default_var_name(node)
        nlags = int(node.get("properties", {}).get("nlags", 20))
        method = str(node.get("properties", {}).get("method", "yw"))
        add("# Partial Auto-correlation")
        add(f"_s_{name} = np.asarray({series}).ravel()")
        add(f"try:")
        add(f"    {name} = sm.tsa.stattools.pacf(_s_{name}, nlags={nlags}, method={json.dumps(method)}) if _s_{name}.size else np.array([])")
        add(f"except Exception:")
        add(f"    {name} = np.array([])")
        return name

    def gen_output(node, inputs):
        in_name = inputs[0][1] if inputs and inputs[0][1] else None
        name = node.get("properties", {}).get("name") or default_var_name(node)
        var = sanitize_name(name)
        if in_name is None:
            add(f"{var} = None")
        else:
            add(f"{var} = {in_name}")
        return var

    # --- New Nodes ---

    def gen_normalize(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        name = default_var_name(node)
        method = node.get("properties", {}).get("method", "minmax")
        add(f"# Data Normalization ({method})")
        if method == "zscore":
            add(f"{name} = ({data} - np.mean({data}, axis=0)) / (np.std({data}, axis=0) + 1e-10)")
        else:
            add(f"{name} = ({data} - np.min({data}, axis=0)) / (np.max({data}, axis=0) - np.min({data}, axis=0) + 1e-10)")
        return name

    def gen_train_test_split(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        y = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        test_size = node.get("properties", {}).get("test_size", 0.2)
        random_state = node.get("properties", {}).get("random_state", 42)
        add(f"# Train Test Split")
        add(f"from sklearn.model_selection import train_test_split")
        add(f"try:")
        add(f"    {name}_X_train, {name}_X_test, {name}_y_train, {name}_y_test = train_test_split({X}, {y}, test_size={test_size}, random_state={random_state})")
        add(f"except Exception:")
        add(f"    {name}_X_train, {name}_X_test, {name}_y_train, {name}_y_test = np.array([]), np.array([]), np.array([]), np.array([])")
        return {0: f"{name}_X_train", 1: f"{name}_X_test", 2: f"{name}_y_train", 3: f"{name}_y_test"}

    def gen_plot_line(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        Y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        title = node.get("properties", {}).get("title", "Line Plot")
        add(f"# Plot Line")
        add(f"plt.figure()")
        add(f"if {X} is not None and {Y} is not None:")
        add(f"    plt.plot({X}, {Y})")
        add(f"elif {Y} is not None:")
        add(f"    plt.plot({Y})")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_plot_scatter(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        Y = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        title = node.get("properties", {}).get("title", "Scatter Plot")
        add(f"# Plot Scatter")
        add(f"plt.figure()")
        add(f"if {X} is not None and {Y} is not None:")
        add(f"    plt.scatter({X}, {Y})")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_ode_solver(node, inputs):
        y0 = inputs[0][1] if len(inputs) > 0 else "[1.0]"
        t = inputs[1][1] if len(inputs) > 1 else "np.linspace(0, 10, 101)"
        name = default_var_name(node)
        func_str = node.get("properties", {}).get("dydt", "y")
        add(f"# ODE Solver")
        add(f"from scipy.integrate import odeint")
        add(f"def ode_func_{name}(y, t):")
        add(f"    # User defined dy/dt")
        add(f"    return {func_str}")
        add(f"try:")
        add(f"    {name} = odeint(ode_func_{name}, {y0}, {t})")
        add(f"except Exception: {name} = []")
        return name

    def gen_ttest(node, inputs):
        a = inputs[0][1] if len(inputs) > 0 else "[]"
        b = inputs[1][1] if len(inputs) > 1 else "[]"
        name = default_var_name(node)
        add(f"# T-Test")
        add(f"try:")
        add(f"    _, {name} = scipy.stats.ttest_ind({a}, {b})")
        add(f"except Exception: {name} = 1.0")
        return name

    # --- New Nodes (MATLAB/LINGO inspired) ---

    def gen_solve_linear(node, inputs):
        A = inputs[0][1] if len(inputs) > 0 else "np.eye(2)"
        b = inputs[1][1] if len(inputs) > 1 else "np.ones(2)"
        name = default_var_name(node)
        add(f"# Solve Linear System Ax = b")
        add(f"try:")
        add(f"    {name} = np.linalg.solve({A}, {b})")
        add(f"except np.linalg.LinAlgError:")
        add(f"    {name} = np.zeros_like({b})")
        return name

    def gen_eigen(node, inputs):
        A = inputs[0][1] if len(inputs) > 0 else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Eigenvalues and Eigenvectors")
        add(f"try:")
        add(f"    {name}_vals, {name}_vecs = np.linalg.eig({A})")
        add(f"except np.linalg.LinAlgError:")
        add(f"    {name}_vals, {name}_vecs = np.array([]), np.array([])")
        return {0: f"{name}_vals", 1: f"{name}_vecs"}

    def gen_fft(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        name = default_var_name(node)
        add(f"# FFT")
        add(f"{name} = np.fft.fft({X}) if {X}.size else np.array([])")
        return name

    def gen_nlp(node, inputs):
        # General nonlinear programming with optional bounds, constraint text, variable names and max/min sense
        import re
        x0 = inputs[0][1] if len(inputs) > 0 else "np.array([0,0])"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        obj_str = props.get("objective", "x[0]**2 + x[1]**2")
        method = props.get("method", "SLSQP")
        constraints_prop = str(props.get("constraints", "") or "")
        bounds_prop = str(props.get("bounds", "") or "")
        variables_prop = str(props.get("variables", "") or "")
        sense = str(props.get("sense", "min")).lower()

        # Parse variable names, auto-detect from expressions if not provided
        var_names = [sanitize_name(v.strip()) for v in re.split(r"[,\\s]+", variables_prop) if v.strip()]
        if not var_names:
            seen: List[str] = []
            for v in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f"{obj_str}\n{constraints_prop}"):
                v_norm = sanitize_name(v)
                if v_norm and v_norm not in seen and not v_norm.isdigit():
                    seen.append(v_norm)
            var_names = seen

        # Parse bounds "0,10;0,inf" or "x1:0,10"
        bounds_list = []
        for line in re.split(r"[;\\n]", bounds_prop):
            if not line.strip():
                continue
            entry = line.strip()
            vname = None
            if ":" in entry:
                vpart, entry = entry.split(":", 1)
                vname = sanitize_name(vpart.strip())
            parts = [p.strip() for p in entry.split(",") if p.strip()]
            lb = None
            ub = None
            if parts:
                try:
                    lb = float(parts[0]) if parts[0].lower() not in ["none", "inf", "+inf"] else None
                except Exception:
                    lb = None
            if len(parts) > 1:
                try:
                    ub = float(parts[1]) if parts[1].lower() not in ["none", "inf", "+inf"] else None
                except Exception:
                    ub = None
            bounds_list.append((vname, lb, ub))

        # Parsed constraints -> list of (lhs, op, rhs, label)
        parsed_constraints: List[Tuple[str, str, str, str]] = []
        for line in re.split(r"[;\\n]", constraints_prop):
            ln = line.strip()
            if not ln:
                continue
            label = None
            if ":" in ln:
                label_part, ln = ln.split(":", 1)
                if label_part.strip():
                    label = sanitize_name(label_part.strip())
                ln = ln.strip()
            m = re.match(r"(.+?)(<=|>=|=)(.+)")
            if not m:
                continue
            lhs, op, rhs = m.groups()
            parsed_constraints.append((lhs.strip(), op, rhs.strip(), label or ""))

        var_names_literal = var_names
        bounds_literal = bounds_list
        constraint_labels = [pc[3] for pc in parsed_constraints]

        add(f"# Non-linear Programming (bounds + constraints + sense)")
        add(f"_x0_{name} = np.atleast_1d(np.asarray({x0} if {x0} is not None else np.zeros(2), dtype=float))")
        add(f"_n_{name} = _x0_{name}.size if _x0_{name}.size else 2")
        add(f"_var_names_{name} = {var_names_literal!r}")
        add(f"if not _var_names_{name} or len(_var_names_{name}) != _n_{name}:")
        add(f"    _var_names_{name} = [f'x{{i}}' for i in range(_n_{name})]")
        add(f"_sense_{name} = '{sense}'")

        add(f"_bounds_raw_{name} = {bounds_literal!r}")
        add(f"_bounds_map_{name} = {{k: (lb, ub) for (k, lb, ub) in _bounds_raw_{name} if k}}")
        add(f"_bounds_{name} = []")
        add(f"for _i in range(_n_{name}):")
        add(f"    _vname = _var_names_{name}[_i] if _i < len(_var_names_{name}) else f'x{{_i}}'")
        add(f"    if _vname in _bounds_map_{name}:")
        add(f"        _bounds_{name}.append(_bounds_map_{name}[_vname])")
        add(f"    elif _i < len(_bounds_raw_{name}):")
        add(f"        _bounds_{name}.append((_bounds_raw_{name}[_i][1], _bounds_raw_{name}[_i][2]))")
        add(f"    else:")
        add(f"        _bounds_{name}.append((None, None))")
        add(f"_bounds_any_{name} = any([(b[0] is not None or b[1] is not None) for b in _bounds_{name}])")
        add(f"_bounds_use_{name} = _bounds_{name} if _bounds_any_{name} else None")

        add(f"def _make_env_{name}(vec):")
        add(f"    _env = {{'x': vec, 'np': np}}")
        add(f"    for _i, _nm in enumerate(_var_names_{name}):")
        add(f"        try: _env[_nm] = vec[_i]")
        add(f"        except Exception: _env[_nm] = 0")
        add(f"    return _env")

        add(f"def obj_{name}(x):")
        add(f"    _env = _make_env_{name}(x)")
        add(f"    _val = eval({obj_str!r}, {{'np': np, '__builtins__': {{}}}}, _env)")
        add(f"    return -_val if _sense_{name} == 'max' else _val")

        add(f"_constraints_{name} = []")
        add(f"_constraint_labels_{name} = {constraint_labels!r}")
        for idx, (lhs, op, rhs, _) in enumerate(parsed_constraints):
            fun_name = f"_con_{name}_{idx}"
            if op == "<=":
                add(f"def {fun_name}(x):")
                add(f"    _env = _make_env_{name}(x)")
                add(f"    return eval({rhs!r}, {{'np': np, '__builtins__': {{}}}}, _env) - eval({lhs!r}, {{'np': np, '__builtins__': {{}}}}, _env)")
                add(f"_constraints_{name}.append({{'type': 'ineq', 'fun': {fun_name}}})")
            elif op == ">=":
                add(f"def {fun_name}(x):")
                add(f"    _env = _make_env_{name}(x)")
                add(f"    return eval({lhs!r}, {{'np': np, '__builtins__': {{}}}}, _env) - eval({rhs!r}, {{'np': np, '__builtins__': {{}}}}, _env)")
                add(f"_constraints_{name}.append({{'type': 'ineq', 'fun': {fun_name}}})")
            else:
                add(f"def {fun_name}(x):")
                add(f"    _env = _make_env_{name}(x)")
                add(f"    return eval({lhs!r}, {{'np': np, '__builtins__': {{}}}}, _env) - eval({rhs!r}, {{'np': np, '__builtins__': {{}}}}, _env)")
                add(f"_constraints_{name}.append({{'type': 'eq', 'fun': {fun_name}}})")

        add(f"res_{name} = scipy.optimize.minimize(obj_{name}, _x0_{name}, method='{method}', bounds=_bounds_use_{name}, constraints=_constraints_{name}) if _x0_{name}.size else None")
        add(f"{name}_x = res_{name}.x if res_{name} is not None and getattr(res_{name}, 'x', None) is not None else _x0_{name}")
        add(f"{name}_obj = obj_{name}({name}_x) if res_{name} is not None else None")
        add(f"if _sense_{name} == 'max' and {name}_obj is not None: {name}_obj = -{name}_obj")
        add(f"{name}_status = res_{name}.message if res_{name} is not None else 'no result'")
        add(f"{name}_constraints = []")
        add(f"for _i, _con in enumerate(_constraints_{name}):")
        add(f"    try:")
        add(f"        _val = float(_con['fun']({name}_x))")
        add(f"    except Exception:")
        add(f"        _val = None")
        add(f"    _ctype = _con.get('type', 'ineq')")
        add(f"    _label = _constraint_labels_{name}[_i] if _i < len(_constraint_labels_{name}) else f'c{{_i}}'")
        add(f"    {name}_constraints.append({{'name': _label, 'type': _ctype, 'residual': _val}})")

        return {0: f"{name}_x", 1: f"{name}_obj", 2: f"{name}_status", 3: f"{name}_constraints", 4: f"_var_names_{name}"}

    def gen_chisquare(node, inputs):
        obs = inputs[0][1] if len(inputs) > 0 else "[]"
        exp = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        add(f"# Chi-Square Test")
        add(f"try:")
        add(f"    _, {name} = scipy.stats.chisquare({obs}, f_exp={exp})")
        add(f"except Exception: {name} = 1.0")
        return name

    def gen_signal_filter(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "[]"
        name = default_var_name(node)
        order = node.get("properties", {}).get("order", 4)
        cutoff = node.get("properties", {}).get("cutoff", 0.1)
        btype = node.get("properties", {}).get("btype", "low")
        add(f"# Signal Filter (Butterworth)")
        add(f"b_{name}, a_{name} = scipy.signal.butter({order}, {cutoff}, btype='{btype}', analog=False)")
        add(f"{name} = scipy.signal.filtfilt(b_{name}, a_{name}, {data}) if len({data}) > 0 else []")
        return name

    def gen_signal_resample(node, inputs):
        data = inputs[0][1] if inputs else "np.array([])"
        fs_in = inputs[1][1] if len(inputs) > 1 else None
        fs_out = inputs[2][1] if len(inputs) > 2 else None
        name = default_var_name(node)
        axis = int(node.get("properties", {}).get("axis", -1))
        add("# Signal Resample")
        add(f"_x_{name} = np.asarray({data})")
        add(f"_fs_in_{name} = float({fs_in}) if {fs_in} is not None else 1.0")
        add(f"_fs_out_{name} = float({fs_out}) if {fs_out} is not None else 1.0")
        add(f"_target_n_{name} = max(1, int(round(_x_{name}.shape[{axis}] * _fs_out_{name} / _fs_in_{name}))) if _x_{name}.size else 0")
        add(f"{name} = scipy.signal.resample(_x_{name}, _target_n_{name}, axis={axis}) if _target_n_{name} > 0 else np.array([])")
        return name

    def gen_signal_stft(node, inputs):
        data = inputs[0][1] if inputs else "np.array([])"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        fs = float(props.get("fs", 1.0))
        nperseg = int(props.get("nperseg", 256))
        noverlap = int(props.get("noverlap", 128))
        window = props.get("window", "hann") or "hann"
        add("# Short-Time Fourier Transform")
        add(f"_x_{name} = np.asarray({data})")
        add(f"{name}_f, {name}_t, {name}_Z = scipy.signal.stft(_x_{name}, fs={fs}, window={json.dumps(window)}, nperseg={nperseg}, noverlap={noverlap})")
        return {0: f"{name}_f", 1: f"{name}_t", 2: f"{name}_Z"}

    def gen_bandpass_filter(node, inputs):
        data = inputs[0][1] if inputs else "np.array([])"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        fs = float(props.get("fs", 1.0))
        lowcut = float(props.get("lowcut", 0.1))
        highcut = float(props.get("highcut", 0.5))
        order = int(props.get("order", 4))
        btype = props.get("btype", "bandpass") or "bandpass"
        add("# Bandpass Filter")
        add(f"_x_{name} = np.asarray({data})")
        add(f"b_{name}, a_{name} = scipy.signal.butter({order}, [{lowcut}, {highcut}], btype='{btype}', fs={fs})")
        add(f"{name} = scipy.signal.filtfilt(b_{name}, a_{name}, _x_{name}) if _x_{name}.size else np.array([])")
        return name

    def gen_xcorr(node, inputs):
        x = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        y = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        mode = str(node.get("properties", {}).get("mode", "full")) or "full"
        add("# Cross Correlation")
        add(f"_x_{name} = np.asarray({x}).ravel()")
        add(f"_y_{name} = np.asarray({y}).ravel()")
        add(f"{name}_corr = scipy.signal.correlate(_x_{name}, _y_{name}, mode={json.dumps(mode)})")
        add(f"{name}_lags = scipy.signal.correlation_lags(len(_x_{name}), len(_y_{name}), mode={json.dumps(mode)})")
        return {0: f"{name}_corr", 1: f"{name}_lags"}

    def gen_transfer_function(node, inputs):
        num = inputs[0][1] if len(inputs) > 0 else "[]"
        den = inputs[1][1] if len(inputs) > 1 else "[]"
        name = default_var_name(node)
        add("# Transfer Function")
        add(f"_num_{name} = np.asarray({num}, dtype=float)")
        add(f"_den_{name} = np.asarray({den}, dtype=float)")
        add(f"{name} = scipy.signal.TransferFunction(_num_{name}, _den_{name})")
        return name

    def gen_step_response(node, inputs):
        sys_var = inputs[0][1] if inputs else "None"
        name = default_var_name(node)
        T_prop = str(node.get("properties", {}).get("T", "")).strip()
        add("# Step Response")
        add(f"_T_{name} = np.fromstring({json.dumps(T_prop)}, sep=',') if {json.dumps(T_prop)} else None")
        add(f"{name}_t, {name}_y = scipy.signal.step({sys_var}, T=_T_{name}) if {sys_var} is not None else (np.array([]), np.array([]))")
        return {0: f"{name}_t", 1: f"{name}_y"}

    def gen_bode(node, inputs):
        sys_var = inputs[0][1] if inputs else "None"
        name = default_var_name(node)
        add("# Bode Plot Data")
        add(f"{name}_w, {name}_mag, {name}_phase = scipy.signal.bode({sys_var}) if {sys_var} is not None else (np.array([]), np.array([]), np.array([]))")
        return {0: f"{name}_w", 1: f"{name}_mag", 2: f"{name}_phase"}

    def gen_plot_hist(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        name = default_var_name(node)
        bins = node.get("properties", {}).get("bins", 10)
        title = node.get("properties", {}).get("title", "Histogram")
        add(f"# Plot Histogram")
        add(f"plt.figure()")
        add(f"if {X} is not None:")
        add(f"    plt.hist({X}, bins={bins})")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_plot_box(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        name = default_var_name(node)
        title = node.get("properties", {}).get("title", "Box Plot")
        add(f"# Plot Box")
        add(f"plt.figure()")
        add(f"if {X} is not None:")
        add(f"    plt.boxplot({X})")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_plot_heatmap(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        name = default_var_name(node)
        title = node.get("properties", {}).get("title", "Heatmap")
        add(f"# Plot Heatmap")
        add(f"plt.figure()")
        add(f"if {X} is not None:")
        add(f"    plt.imshow({X}, cmap='viridis', aspect='auto')")
        add(f"    plt.colorbar()")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_plot_surface(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        Y = inputs[1][1] if len(inputs) > 1 else "None"
        Z = inputs[2][1] if len(inputs) > 2 else "None"
        name = default_var_name(node)
        title = node.get("properties", {}).get("title", "3D Surface")
        add(f"# Plot 3D Surface")
        add(f"fig = plt.figure()")
        add(f"ax = fig.add_subplot(111, projection='3d')")
        add(f"if {X} is not None and {Y} is not None and {Z} is not None:")
        add(f"    ax.plot_surface({X}, {Y}, {Z}, cmap='viridis')")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_load_csv(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        path = props.get("path", "data.csv")
        header = props.get("header", 0)
        encoding = props.get("encoding", "utf-8")
        usecols_raw = str(props.get("usecols", "") or "")
        output_format = str(props.get("output_format", "matrix")).lower()
        if output_format not in {"matrix", "dataframe", "records"}:
            output_format = "matrix"
        usecols = [c.strip() for c in usecols_raw.split(",") if c.strip()]
        path_literal = json.dumps(path)
        encoding_literal = json.dumps(encoding)
        usecols_literal = f"[{', '.join(repr(c) for c in usecols)}]" if usecols else "None"
        add(f"# Load CSV")
        add(f"try:")
        add(f"    _df_{name} = pd.read_csv({path_literal}, header={header}, encoding={encoding_literal}" + (", usecols=" + usecols_literal if usecols else "") + ")")
        if output_format == "dataframe":
            add(f"    {name} = _df_{name}")
        elif output_format == "records":
            add(f"    {name} = _df_{name}.to_dict(orient='records')")
        else:
            add(f"    {name} = _df_{name}.values")
        add(f"except Exception:")
        if output_format == "dataframe":
            add(f"    {name} = pd.DataFrame()")
        elif output_format == "records":
            add(f"    {name} = []")
        else:
            add(f"    {name} = np.array([])")
        return name

    def gen_load_excel(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        path = props.get("path", "data.xlsx")
        sheet = props.get("sheet", 0)
        usecols_raw = str(props.get("usecols", "") or "")
        output_format = str(props.get("output_format", "matrix")).lower()
        if output_format not in {"matrix", "dataframe", "records"}:
            output_format = "matrix"
        usecols = [c.strip() for c in usecols_raw.split(",") if c.strip()]
        path_literal = json.dumps(path)
        sheet_literal = json.dumps(sheet) if isinstance(sheet, str) else sheet
        usecols_literal = f"[{', '.join(repr(c) for c in usecols)}]" if usecols else "None"
        add(f"# Load Excel")
        add(f"try:")
        add(f"    _df_{name} = pd.read_excel({path_literal}, sheet_name={sheet_literal}" + (", usecols=" + usecols_literal if usecols else "") + ")")
        if output_format == "dataframe":
            add(f"    {name} = _df_{name}")
        elif output_format == "records":
            add(f"    {name} = _df_{name}.to_dict(orient='records')")
        else:
            add(f"    {name} = _df_{name}.values")
        add(f"except Exception:")
        if output_format == "dataframe":
            add(f"    {name} = pd.DataFrame()")
        elif output_format == "records":
            add(f"    {name} = []")
        else:
            add(f"    {name} = np.array([])")
        return name

    def gen_load_excel_adv(node, inputs):
        """Load Excel with multi-row headers and stitched column names."""
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        path = props.get("path", "data.xlsx")
        sheet = props.get("sheet", 0)
        header_rows_raw = str(props.get("header_rows", "")).strip()
        data_start_raw = str(props.get("data_start_row", "")).strip()
        combine_mode = str(props.get("combine_mode", "code+name")).lower()
        city_col = str(props.get("city_column", "city") or "city")
        drop_empty = bool(props.get("drop_empty_cols", True))
        output_format = str(props.get("output_format", "dataframe")).lower()
        if output_format not in {"matrix", "dataframe", "records"}:
            output_format = "dataframe"

        header_rows: List[int] = []
        for part in header_rows_raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                row_idx = int(float(part)) - 1  # user-friendly 1-based -> 0-based
                if row_idx >= 0:
                    header_rows.append(row_idx)
            except Exception:
                continue
        header_rows = sorted(set(header_rows))

        data_start_row = None
        try:
            ds = int(float(data_start_raw))
            if ds >= 1:
                data_start_row = ds - 1  # convert to 0-based
        except Exception:
            data_start_row = None

        path_literal = json.dumps(path)
        sheet_literal = json.dumps(sheet) if isinstance(sheet, str) else sheet
        header_literal = f"[{', '.join(str(h) for h in header_rows)}]"
        combine_literal = json.dumps(combine_mode)
        city_literal = json.dumps(city_col)
        start_literal = "None" if data_start_row is None else str(data_start_row)
        add("# Advanced Excel loader (stitch multi-row headers)")
        add("try:")
        add(f"    _raw_{name} = pd.read_excel({path_literal}, sheet_name={sheet_literal}, header=None)")
        add(f"    _header_rows_{name} = {header_literal}")
        add(f"    _combine_mode_{name} = {combine_literal}")
        add(f"    _header_values_{name} = [_raw_{name}.iloc[r] for r in _header_rows_{name} if r < len(_raw_{name})]")
        add(f"    _ncols_{name} = _raw_{name}.shape[1] if hasattr(_raw_{name}, 'shape') else 0")
        add(f"    _columns_{name} = []")
        add(f"    for _c in range(_ncols_{name}):")
        add(f"        _parts = []")
        add(f"        for _hdr in _header_values_{name}:")
        add(f"            if _c < len(_hdr):")
        add(f"                _val = _hdr.iloc[_c]")
        add(f"                if pd.notna(_val) and str(_val).strip():")
        add(f"                    _parts.append(str(_val).strip())")
        add(f"        _label = ''")
        add(f"        if _combine_mode_{name} == 'code+name':")
        add(f"            if len(_parts) >= 2:")
        add(f"                _label = f\"{{_parts[0]}}_{{_parts[1]}}\"")
        add(f"            elif _parts:")
        add(f"                _label = _parts[0]")
        add(f"        elif _combine_mode_{name} == 'name_only':")
        add(f"            _label = _parts[1] if len(_parts) > 1 else (_parts[0] if _parts else '')")
        add(f"        elif _combine_mode_{name} == 'code_only':")
        add(f"            _label = _parts[0] if _parts else ''")
        add(f"        else:  # first_nonempty")
        add(f"            _label = _parts[0] if _parts else ''")
        add(f"        if not _label:")
        add(f"            _label = f\"col{{_c}}\"")
        add(f"        _columns_{name}.append(_label)")
        add(f"    _start_row_{name} = {start_literal}")
        add(f"    if _start_row_{name} is None:")
        add(f"        _start_row_{name} = max(_header_rows_{name}) + 1 if _header_rows_{name} else 0")
        add(f"    _df_{name} = _raw_{name}.iloc[_start_row_{name}:].copy()")
        add(f"    if len(_columns_{name}) == _df_{name}.shape[1]:")
        add(f"        _df_{name}.columns = _columns_{name}")
        add(f"    if {json.dumps(drop_empty)}:")
        add(f"        _df_{name} = _df_{name}.loc[:, ~_df_{name}.isna().all()]")
        add(f"    if {city_literal} and _df_{name}.shape[1] > 0:")
        add(f"        _first_col = _df_{name}.columns[0]")
        add(f"        _df_{name}.rename(columns={{_first_col: {city_literal}}}, inplace=True)")
        if output_format == "matrix":
            add(f"    {name} = _df_{name}.values")
        elif output_format == "records":
            add(f"    {name} = _df_{name}.to_dict(orient='records')")
        else:
            add(f"    {name} = _df_{name}")
        add(f"except Exception:")
        if output_format == "matrix":
            add(f"    {name} = np.array([])")
        elif output_format == "records":
            add(f"    {name} = []")
        else:
            add(f"    {name} = pd.DataFrame()")
        return name

    def gen_select_column(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        selector = str(props.get("selector", "0"))
        mode = str(props.get("mode", "index")).lower()
        as_array = bool(props.get("as_array", True))
        try:
            col_idx = int(float(selector))
        except Exception:
            col_idx = 0
        selector_literal = json.dumps(selector)
        add(f"# Select Column/Series")
        add(f"_source_{name} = {data}")
        add(f"_df_{name} = _source_{name} if isinstance(_source_{name}, pd.DataFrame) else pd.DataFrame(_source_{name})")
        add(f"try:")
        if mode == "name":
            add(f"    if {selector_literal} in _df_{name}.columns:")
            add(f"        _sel_{name} = _df_{name}[{selector_literal}]")
            add(f"    else:")
            add(f"        _sel_{name} = _df_{name}.iloc[:, {col_idx}]")
        else:
            add(f"    _sel_{name} = _df_{name}.iloc[:, {col_idx}]")
        if as_array:
            add(f"    {name} = _sel_{name}.values")
        else:
            add(f"    {name} = _sel_{name}")
        add(f"except Exception:")
        if as_array:
            add(f"    {name} = np.array([])")
        else:
            add(f"    {name} = pd.Series(dtype=float)")
        return name

    def gen_filter_rows(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        condition = str(props.get("condition", "")).strip()
        reset_index = bool(props.get("reset_index", True))
        output_format = str(props.get("output_format", "dataframe")).lower()
        if output_format not in {"matrix", "dataframe"}:
            output_format = "dataframe"
        condition_literal = json.dumps(condition)
        add(f"# Filter Rows")
        add(f"_source_{name} = {data}")
        add(f"_df_{name} = _source_{name} if isinstance(_source_{name}, pd.DataFrame) else pd.DataFrame(_source_{name})")
        add(f"try:")
        if condition:
            add(f"    _filtered_{name} = _df_{name}.query({condition_literal})")
        else:
            add(f"    _filtered_{name} = _df_{name}.copy()")
        if reset_index:
            add(f"    _filtered_{name} = _filtered_{name}.reset_index(drop=True)")
        add(f"except Exception:")
        add(f"    _filtered_{name} = _df_{name}.copy()")
        if output_format == "matrix":
            add(f"{name} = _filtered_{name}.values")
        else:
            add(f"{name} = _filtered_{name}")
        return name

    def gen_group_aggregate(node, inputs):
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        group_raw = str(props.get("group_by", ""))
        agg_raw = str(props.get("aggregations", ""))
        reset_index = bool(props.get("reset_index", True))
        flatten_cols = bool(props.get("flatten_columns", True))
        dropna = bool(props.get("dropna", True))
        output_format = str(props.get("output_format", "dataframe")).lower()
        if output_format not in {"matrix", "dataframe"}:
            output_format = "dataframe"
        group_cols = [c.strip() for c in group_raw.split(",") if c.strip()]
        agg_spec = {}
        for part in agg_raw.split(","):
            if ":" not in part:
                continue
            col, func = part.split(":", 1)
            funcs = [f.strip() for f in func.split("|") if f.strip()]
            if not funcs:
                funcs = ["sum"]
            agg_spec[col.strip()] = funcs if len(funcs) > 1 else funcs[0]
        group_literal = f"[{', '.join(repr(c) for c in group_cols)}]"
        agg_literal = "{" + ", ".join(f"{repr(col)}: {repr(funcs)}" for col, funcs in agg_spec.items()) + "}"
        add(f"# Group & Aggregate")
        add(f"_source_{name} = {data}")
        add(f"_df_{name} = _source_{name} if isinstance(_source_{name}, pd.DataFrame) else pd.DataFrame(_source_{name})")
        add(f"_group_cols_{name} = {group_literal}")
        add(f"_agg_map_{name} = {agg_literal}")
        add(f"if _group_cols_{name} and _agg_map_{name}:")
        add(f"    _agg_{name} = _df_{name}.groupby(_group_cols_{name}, dropna={dropna}).agg(_agg_map_{name})")
        if reset_index:
            add(f"    _agg_{name} = _agg_{name}.reset_index()")
        if flatten_cols:
            add(f"    if isinstance(_agg_{name}.columns, pd.MultiIndex):")
            add(f"        _agg_{name}.columns = ['_'.join([str(x) for x in tpl if str(x)]) for tpl in _agg_{name}.columns.values]")
        add(f"else:")
        add(f"    _agg_{name} = _df_{name}.copy()")
        if output_format == "matrix":
            add(f"{name} = _agg_{name}.values")
        else:
            add(f"{name} = _agg_{name}")
        return name

    def gen_pivot_table(node, inputs):
        data = inputs[0][1] if inputs else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        index_cols = [c.strip() for c in str(props.get("index", "")).split(",") if c.strip()]
        column_cols = [c.strip() for c in str(props.get("columns", "")).split(",") if c.strip()]
        values_col = str(props.get("values", "")).strip()
        aggfunc = props.get("aggfunc", "mean") or "mean"
        fill_value = props.get("fill_value", "")
        margins = bool(props.get("margins", False))

        index_literal = "[" + ", ".join(repr(c) for c in index_cols) + "]" if index_cols else "None"
        column_literal = "[" + ", ".join(repr(c) for c in column_cols) + "]" if column_cols else "None"
        values_literal = repr(values_col) if values_col else "None"
        fill_literal = "None" if fill_value == "" else repr(fill_value)

        add("# Pivot Table")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"{name} = pd.pivot_table(_df_{name}, index={index_literal}, columns={column_literal}, values={values_literal}, aggfunc={repr(aggfunc)}, fill_value={fill_literal}, margins={margins})")
        return name

    def gen_conditional_column(node, inputs):
        data = inputs[0][1] if inputs else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        condition = str(props.get("condition", ""))
        true_value = props.get("true_value", 1)
        false_value = props.get("false_value", 0)
        output_col = str(props.get("output_column", "flag")) or "flag"

        cond_literal = json.dumps(condition)
        out_literal = json.dumps(output_col)

        add("# Conditional Column")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_result_{name} = _df_{name}.copy()")
        add(f"try:")
        add(f"    _mask_{name} = _result_{name}.eval({cond_literal})")
        add(f"except Exception:")
        add(f"    _mask_{name} = False")
        add(f"_result_{name}[{out_literal}] = np.where(_mask_{name}, {json.dumps(true_value)}, {json.dumps(false_value)})")
        add(f"{name} = _result_{name}")
        return name

    def gen_describe(node, inputs):
        """Quickly summarize a dataset to reduce ad-hoc inspection code."""
        data = inputs[0][1] if inputs else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        include = str(props.get("include", "all")) or "all"
        percentiles_raw = str(props.get("percentiles", "0.25,0.5,0.75"))

        percentiles: List[float] = []
        for p in percentiles_raw.split(","):
            try:
                val = float(p.strip())
                if 0 < val < 1:
                    percentiles.append(val)
            except Exception:
                continue
        if not percentiles:
            percentiles = [0.25, 0.5, 0.75]

        include_literal = json.dumps(include)

        add("# Describe DataFrame")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_percentiles_{name} = {percentiles}")
        add(f"try:")
        add(f"    {name} = _df_{name}.describe(include={include_literal}, percentiles=_percentiles_{name})")
        add(f"except Exception:")
        add(f"    {name} = _df_{name}")
        return name

    def gen_weighted_score(node, inputs):
        """Normalize indicators (pos/neg) and compute weighted score."""
        data = inputs[0][1] if inputs else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        spec_raw = str(props.get("indicators", "")).strip()
        normalize = str(props.get("normalize", "minmax")).lower()
        add_norm = bool(props.get("add_normalized_cols", True))
        score_col = str(props.get("score_column", "score") or "score")
        output_format = str(props.get("output_format", "dataframe")).lower()
        if output_format not in {"matrix", "dataframe", "records"}:
            output_format = "dataframe"

        specs = []
        for part in spec_raw.split(","):
            part = part.strip()
            if not part:
                continue
            tokens = [t.strip() for t in part.split(":")]
            if len(tokens) < 2:
                continue
            col = tokens[0]
            direction = "N" if tokens[1].upper().startswith("N") else "P"
            try:
                weight = float(tokens[2]) if len(tokens) > 2 else 1.0
            except Exception:
                weight = 1.0
            specs.append({"col": col, "direction": direction, "weight": weight})

        specs_literal = json.dumps(specs, ensure_ascii=False)
        add("# Weighted normalization & scoring")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_specs_{name} = {specs_literal}")
        add(f"_result_{name} = _df_{name}.copy()")
        add(f"_result_{name}[{json.dumps(score_col)}] = 0.0")
        add(f"for _spec in _specs_{name}:")
        add(f"    _col = _spec.get('col')")
        add(f"    if _col not in _result_{name}.columns:")
        add(f"        continue")
        add(f"    _series = pd.to_numeric(_result_{name}[_col], errors='coerce')")
        if normalize == "zscore":
            add(f"    _norm = (_series - _series.mean()) / (_series.std() + 1e-10)")
        else:
            add(f"    _norm = (_series - _series.min()) / ((_series.max() - _series.min()) + 1e-10)")
        add(f"    if _spec.get('direction') == 'N':")
        add(f"        _norm = 1 - _norm")
        if add_norm:
            add(f"    _result_{name}[f\"{{_col}}_norm\"] = _norm")
        add(f"    _result_{name}[{json.dumps(score_col)}] = _result_{name}[{json.dumps(score_col)}] + _norm.fillna(0) * _spec.get('weight', 1.0)")
        if output_format == "matrix":
            add(f"{name} = _result_{name}.values")
        elif output_format == "records":
            add(f"{name} = _result_{name}.to_dict(orient='records')")
        else:
            add(f"{name} = _result_{name}")
        return name

    def gen_indicator_dict(node, inputs):
        """Materialize an indicator dictionary for transparency."""
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        spec_raw = str(props.get("indicators", "")).strip()
        desc_raw = str(props.get("descriptions", "")).strip()

        specs = []
        for part in spec_raw.split(","):
            part = part.strip()
            if not part:
                continue
            tokens = [t.strip() for t in part.split(":")]
            if len(tokens) < 2:
                continue
            col = tokens[0]
            direction = "N" if tokens[1].upper().startswith("N") else "P"
            try:
                weight = float(tokens[2]) if len(tokens) > 2 else 1.0
            except Exception:
                weight = 1.0
            specs.append({"col": col, "direction": direction, "weight": weight})

        desc_map = {}
        for part in desc_raw.split(";"):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            if k:
                desc_map[k] = v

        rows = []
        for spec in specs:
            rows.append({
                "indicator": spec["col"],
                "direction": spec["direction"],
                "weight": spec["weight"],
                "description": desc_map.get(spec["col"], "")
            })

        rows_literal = json.dumps(rows, ensure_ascii=False)
        add("# Indicator dictionary (for documentation)")
        add(f"{name} = pd.DataFrame({rows_literal})")
        return name

    def gen_rolling_window(node, inputs):
        """Calculate rolling window statistics"""
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        column = str(props.get("column", ""))
        window = int(props.get("window", 3))
        operation = str(props.get("operation", "mean")).lower()
        groupby_cols = str(props.get("groupby", ""))
        output_col = str(props.get("output_column", "")) or f"{column}_{operation}_{window}"
        min_periods = int(props.get("min_periods", 1))
        
        add(f"# Rolling Window: {operation} over {window} periods")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_result_{name} = _df_{name}.copy()")
        
        col_literal = json.dumps(column)
        out_literal = json.dumps(output_col)
        
        if groupby_cols:
            groupby_list = [c.strip() for c in groupby_cols.split(",") if c.strip()]
            groupby_literal = json.dumps(groupby_list)
            add(f"_rolling_{name} = _result_{name}.groupby({groupby_literal})[{col_literal}].rolling({window}, min_periods={min_periods})")
        else:
            add(f"_rolling_{name} = _result_{name}[{col_literal}].rolling({window}, min_periods={min_periods})")
        
        if operation == "mean":
            add(f"_rolled_{name} = _rolling_{name}.mean()")
        elif operation == "sum":
            add(f"_rolled_{name} = _rolling_{name}.sum()")
        elif operation == "std":
            add(f"_rolled_{name} = _rolling_{name}.std()")
        elif operation == "min":
            add(f"_rolled_{name} = _rolling_{name}.min()")
        elif operation == "max":
            add(f"_rolled_{name} = _rolling_{name}.max()")
        elif operation == "median":
            add(f"_rolled_{name} = _rolling_{name}.median()")
        else:
            add(f"_rolled_{name} = _rolling_{name}.mean()")
        
        if groupby_cols:
            add(f"_result_{name}[{out_literal}] = _rolled_{name}.reset_index(level=0, drop=True)")
        else:
            add(f"_result_{name}[{out_literal}] = _rolled_{name}")
        
        add(f"{name} = _result_{name}")
        return name

    def gen_transform_column(node, inputs):
        """Transform a column with various operations"""
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        column = str(props.get("column", ""))
        operation = str(props.get("operation", "identity")).lower()
        output_col = str(props.get("output_column", "")) or f"{column}_{operation}"
        groupby_cols = str(props.get("groupby", ""))
        
        add(f"# Transform Column: {operation}")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_result_{name} = _df_{name}.copy()")
        
        col_literal = json.dumps(column)
        out_literal = json.dumps(output_col)
        
        # Helper to apply operation with optional groupby
        def apply_op(op_code):
            if groupby_cols:
                groupby_list = [c.strip() for c in groupby_cols.split(",") if c.strip()]
                groupby_literal = json.dumps(groupby_list)
                return f"_result_{name}.groupby({groupby_literal})[{col_literal}].{op_code}"
            else:
                return f"_result_{name}[{col_literal}].{op_code}"

        if operation == "fillna":
            fill_value = props.get("fill_value", 0)
            # fillna doesn't really need groupby usually, but for consistency... 
            # actually fillna is element-wise. Groupby fillna (ffill/bfill) is different.
            # For simple value fill, no groupby needed.
            add(f"_result_{name}[{out_literal}] = _result_{name}[{col_literal}].fillna({fill_value})")
        elif operation == "diff":
            periods = int(props.get("periods", 1))
            add(f"_result_{name}[{out_literal}] = {apply_op(f'diff({periods})')}.fillna(0)")
        elif operation == "pct_change":
            periods = int(props.get("periods", 1))
            add(f"_result_{name}[{out_literal}] = {apply_op(f'pct_change({periods})')}.fillna(0)")
        elif operation == "shift":
            periods = int(props.get("periods", 1))
            add(f"_result_{name}[{out_literal}] = {apply_op(f'shift({periods})')}")
        elif operation == "cumsum":
            add(f"_result_{name}[{out_literal}] = {apply_op('cumsum()')}")
        elif operation == "log":
            add(f"_result_{name}[{out_literal}] = np.log(_result_{name}[{col_literal}] + 1e-10)")
        elif operation == "sqrt":
            add(f"_result_{name}[{out_literal}] = np.sqrt(np.abs(_result_{name}[{col_literal}]))")
        elif operation == "abs":
            add(f"_result_{name}[{out_literal}] = np.abs(_result_{name}[{col_literal}])")
        elif operation == "round":
            decimals = int(props.get("decimals", 0))
            add(f"_result_{name}[{out_literal}] = _result_{name}[{col_literal}].round({decimals})")
        else:
            add(f"_result_{name}[{out_literal}] = _result_{name}[{col_literal}]")
        
        add(f"{name} = _result_{name}")
        return name

    def gen_merge_dataframes(node, inputs):
        """Merge two dataframes"""
        left = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        right = inputs[1][1] if len(inputs) > 1 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        how = str(props.get("how", "inner"))
        left_on = str(props.get("left_on", ""))
        right_on = str(props.get("right_on", ""))
        on = str(props.get("on", ""))
        
        add(f"# Merge DataFrames ({how} join)")
        add(f"_left_{name} = {left} if isinstance({left}, pd.DataFrame) else pd.DataFrame({left})")
        add(f"_right_{name} = {right} if isinstance({right}, pd.DataFrame) else pd.DataFrame({right})")
        
        if on:
            on_literal = json.dumps(on)
            add(f"{name} = pd.merge(_left_{name}, _right_{name}, on={on_literal}, how='{how}')")
        elif left_on and right_on:
            left_on_literal = json.dumps(left_on)
            right_on_literal = json.dumps(right_on)
            add(f"{name} = pd.merge(_left_{name}, _right_{name}, left_on={left_on_literal}, right_on={right_on_literal}, how='{how}')")
        else:
            add(f"{name} = pd.merge(_left_{name}, _right_{name}, how='{how}')")
        
        return name

    def gen_time_features(node, inputs):
        """Extract time-based features from a date column"""
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        date_column = str(props.get("date_column", "Year"))
        features = str(props.get("features", "year,month,dayofweek"))
        
        add(f"# Extract Time Features")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_result_{name} = _df_{name}.copy()")
        
        date_col_literal = json.dumps(date_column)
        add(f"_date_series_{name} = pd.to_datetime(_result_{name}[{date_col_literal}], errors='coerce')")
        
        feature_list = [f.strip().lower() for f in features.split(",") if f.strip()]
        for feat in feature_list:
            if feat == "year":
                add(f"_result_{name}['year'] = _date_series_{name}.dt.year")
            elif feat == "month":
                add(f"_result_{name}['month'] = _date_series_{name}.dt.month")
            elif feat == "day":
                add(f"_result_{name}['day'] = _date_series_{name}.dt.day")
            elif feat == "dayofweek":
                add(f"_result_{name}['dayofweek'] = _date_series_{name}.dt.dayofweek")
            elif feat == "quarter":
                add(f"_result_{name}['quarter'] = _date_series_{name}.dt.quarter")
            elif feat == "dayofyear":
                add(f"_result_{name}['dayofyear'] = _date_series_{name}.dt.dayofyear")
            elif feat == "weekofyear":
                add(f"_result_{name}['weekofyear'] = _date_series_{name}.dt.isocalendar().week")
        
        add(f"{name} = _result_{name}")
        return name

    def gen_create_dummy(node, inputs):
        """Generates dummy variables (one-hot encoding or binary flag)."""
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        column = str(props.get("column", ""))
        mode = str(props.get("mode", "onehot")).lower()
        value = str(props.get("value", "")) # For binary mode
        output_col = str(props.get("output_column", "")) # For binary mode
        prefix = str(props.get("prefix", "")) or column

        add(f"# Create Dummy Variables: mode={mode}")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_result_{name} = _df_{name}.copy()")
        
        col_literal = json.dumps(column)
        
        add(f"if {col_literal} in _result_{name}.columns:")
        if mode == "binary":
            out_literal = json.dumps(output_col or f"{column}_{value}_flag")
            value_literal = json.dumps(value)
            add(f"    _result_{name}[{out_literal}] = (_result_{name}[{col_literal}].astype(str) == {value_literal}).astype(int)")
        else: # onehot
            prefix_literal = json.dumps(prefix)
            add(f"    _dummies_{name} = pd.get_dummies(_result_{name}[{col_literal}], prefix={prefix_literal}, dtype=int)")
            add(f"    _result_{name} = pd.concat([_result_{name}, _dummies_{name}], axis=1)")
            # Drop original column after encoding
            add(f"    _result_{name} = _result_{name}.drop(columns=[{col_literal}])")
        add(f"else:")
        add(f"    print(f'Warning: Column \"{column}\" not found for dummy creation.')")

        add(f"{name} = _result_{name}")
        return name

    def gen_map_values(node, inputs):
        """Maps values in a column based on a dictionary."""
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        column = str(props.get("column", ""))
        mapping_dict_str = str(props.get("mapping_dict", "{}"))
        default_value = props.get("default_value", None)
        output_col = str(props.get("output_column", "")) or f"{column}_mapped"

        add(f"# Map Values in Column")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_result_{name} = _df_{name}.copy()")
        
        col_literal = json.dumps(column)
        out_literal = json.dumps(output_col)
        
        add(f"if {col_literal} in _result_{name}.columns:")
        add(f"    try:")
        add(f"        mapping_dict_{name} = json.loads('{mapping_dict_str}')")
        add(f"        # Ensure keys are of the same type as the column if possible")
        add(f"        col_type_{name} = _result_{name}[{col_literal}].dtype")
        add(f"        if np.issubdtype(col_type_{name}, np.number):")
        add(f"            mapping_dict_{name} = {{float(k): v for k, v in mapping_dict_{name}.items()}}")
        add(f"        mapped_series_{name} = _result_{name}[{col_literal}].map(mapping_dict_{name})")
        if default_value is not None:
            default_literal = json.dumps(default_value)
            add(f"        _result_{name}[{out_literal}] = mapped_series_{name}.fillna({default_literal})")
        else:
            add(f"        # If no default, keep original values for non-mapped items")
            add(f"        _result_{name}[{out_literal}] = mapped_series_{name}.fillna(_result_{name}[{col_literal}])")
        add(f"    except Exception as e:")
        add(f"        print(f'Map Values node failed: {{e}}')")
        add(f"        _result_{name}[{out_literal}] = _result_{name}[{col_literal}]")
        add(f"else:")
        add(f"    print(f'Warning: Column \"{column}\" not found for mapping.')")

        add(f"{name} = _result_{name}")
        return name

    def gen_explode_column(node, inputs):
        """Expands list-like entries in a column into separate rows."""
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        column = str(props.get("column", ""))
        output_col = str(props.get("output_column", "")).strip()
        ignore_index = bool(props.get("ignore_index", True))

        add(f"# Explode Column")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_result_{name} = _df_{name}.copy()")

        col_literal = json.dumps(column)
        out_literal = json.dumps(output_col) if output_col else col_literal

        add(f"if {col_literal} in _result_{name}.columns:")
        if output_col and output_col != column:
            add(f"    _result_{name}[{out_literal}] = _result_{name}[{col_literal}]")
        add(f"    try:")
        add(f"        _result_{name} = _result_{name}.explode({out_literal}, ignore_index={ignore_index})")
        add(f"    except Exception as e:")
        add(f"        print(f'Explode column failed for {column}: {{e}}')")
        add(f"else:")
        add(f"    print(f'Warning: Column \"{column}\" not found for explode.')")

        add(f"{name} = _result_{name}")
        return name

    def gen_expression(node, inputs):
        """Evaluates a pandas expression on a DataFrame."""
        data = inputs[0][1] if len(inputs) > 0 else "pd.DataFrame()"
        name = default_var_name(node)
        props = node.get("properties", {}) or {}
        expression = str(props.get("expression", "1")).strip()
        output_col = str(props.get("output_column", "result")).strip()

        add(f"# Evaluate Expression: '{output_col}' = {expression}")
        add(f"_df_{name} = {data} if isinstance({data}, pd.DataFrame) else pd.DataFrame({data})")
        add(f"_result_{name} = _df_{name}.copy()")
        
        expr_literal = json.dumps(expression)
        out_literal = json.dumps(output_col)

        add(f"try:")
        # The result of eval is assigned to the new column
        add(f"    _result_{name}[{out_literal}] = _result_{name}.eval({expr_literal}, engine='python')")
        add(f"except Exception as e:")
        add(f"    print(f'Expression node failed for expression \"{expression}\": {{e}}')")
        add(f"    _result_{name}[{out_literal}] = np.nan")

        add(f"{name} = _result_{name}")
        return name

    def gen_custom_python(node, inputs):
        name = default_var_name(node)
        code = node.get("properties", {}).get("code", "")
        input_names_str = node.get("properties", {}).get("inputs", "")
        output_names_str = node.get("properties", {}).get("outputs", "")
        
        input_names = [s.strip() for s in input_names_str.split(",") if s.strip()]
        output_names = [s.strip() for s in output_names_str.split(",") if s.strip()]
        
        add(f"# Custom Python Script Node: {name}")
        add(f"def custom_func_{name}(" + ", ".join(input_names) + "):")
        
        # Indent user code
        for line in code.splitlines():
            add(f"    {line}")
            
        # Return outputs
        if output_names:
            add(f"    return " + ", ".join(output_names))
        else:
            add(f"    return None")
            
        # Call function
        args = []
        for i, in_name in enumerate(input_names):
            if i < len(inputs) and inputs[i][1] is not None:
                args.append(inputs[i][1])
            else:
                args.append("None")
                
        add(f"_res_{name} = custom_func_{name}(" + ", ".join(args) + ")")
        
        # Unpack results
        if len(output_names) > 1:
            # If multiple outputs, we assume the function returns a tuple
            # We need to map slot index to the unpacked variable
            # But wait, `resolve_input_vars` expects `varmap` to return a name or a dict.
            # Let's store the result tuple in a variable, and return a dict that accesses it.
            # Actually, let's unpack it right here for clarity in generated code.
            ret_dict = {}
            for i, out_name in enumerate(output_names):
                var_name = f"{name}_{out_name}"
                add(f"{var_name} = _res_{name}[{i}] if _res_{name} is not None and len(_res_{name}) > {i} else None")
                ret_dict[i] = var_name
            return ret_dict
        elif len(output_names) == 1:
            add(f"{name} = _res_{name}")
            return name
        else:
            return name


    def gen_subgraph(node, inputs):
        # Use node title for function name if available
        title = node.get("title", "subgraph")
        fname = f"{sanitize_name(title)}_{node['id']}"
        
        inner_data = node.get("subgraph")
        if not inner_data:
            add(f"# Empty subgraph {fname}")
            return "None"
            
        # Analyze inner inputs/outputs
        inner_nodes = inner_data.get("nodes", [])
        
        # Map inner GraphInput/GraphOutput nodes by name
        # GraphInput/Output nodes store their name in properties['name']
        inner_input_nodes = {n.get("properties", {}).get("name", ""): n for n in inner_nodes if n["type"] == "graph/input"}
        inner_output_nodes = {n.get("properties", {}).get("name", ""): n for n in inner_nodes if n["type"] == "graph/output"}
        
        # Prepare arguments based on the OUTER node's inputs
        # The outer node inputs determine the function signature order
        arg_names = []
        inner_varmap_init = {}
        used_arg_names = set()
        
        # inputs is a list of (slot_name, source_var_name)
        # We iterate over it to define arguments
        for i, (slot_name, src_var) in enumerate(inputs):
            # Try to use the slot name as argument name
            base_name = sanitize_name(slot_name) if slot_name and slot_name.strip() else f"in_{i}"
            arg_name = base_name
            
            # Handle duplicates
            k = 1
            while arg_name in used_arg_names:
                arg_name = f"{base_name}_{k}"
                k += 1
            
            used_arg_names.add(arg_name)
            arg_names.append(arg_name)
            
            # Find corresponding inner node
            # LiteGraph ensures slot name matches inner node name
            inner_node = inner_input_nodes.get(slot_name)
            if inner_node:
                inner_varmap_init[inner_node["id"]] = arg_name
            else:
                # Fallback: try to find by order if names don't match?
                # Or maybe the slot name is different from property name?
                # LiteGraph: Subgraph inputs are synced with GraphInput nodes.
                # If name mismatch, we might have an issue.
                # Let's assume name match for now.
                pass

        add(f"def {fname}({', '.join(arg_names)}):")
        
        # Recursive generation
        inner_lines, inner_final_varmap = generate_scope(inner_data, inner_varmap_init)
        
        for line in inner_lines:
            add(f"    {line}")
            
        # Return statement
        # We need to return values corresponding to the OUTER node's outputs
        outer_outputs = node.get("outputs", [])
        ret_vars = []
        
        for out_slot in outer_outputs:
            slot_name = out_slot.get("name")
            inner_node = inner_output_nodes.get(slot_name)
            if inner_node:
                # The value to return is the variable connected to the GraphOutput node
                val = inner_final_varmap.get(inner_node["id"])
                ret_vars.append(val if val else "None")
            else:
                ret_vars.append("None")
            
        if ret_vars:
            add(f"    return {', '.join(ret_vars)}")
        else:
            add(f"    return None")
            
        # Call the function
        args = [inp[1] if inp[1] else "None" for inp in inputs]
        add(f"_res_{fname} = {fname}({', '.join(args)})")
        
        # Unpack results
        if len(outer_outputs) > 1:
            ret_dict = {}
            for i, out_slot in enumerate(outer_outputs):
                var_name = f"{fname}_{sanitize_name(out_slot.get('name', f'out{i}'))}"
                add(f"{var_name} = _res_{fname}[{i}] if _res_{fname} is not None and len(_res_{fname}) > {i} else None")
                ret_dict[i] = var_name
            return ret_dict
        elif len(outer_outputs) == 1:
            add(f"{fname} = _res_{fname}")
            return fname
        else:
            return fname

    def gen_graph_input(node, inputs):
        # This node represents an argument to the subgraph function.
        # The variable name is already in varmap (populated before recursion).
        # We just return it.
        # But wait, generators return a NEW variable name usually.
        # Here we want to return the EXISTING name.
        # The generate_scope loop assigns: varmap[nid] = gen(...)
        # So we must return the argument name here.
        # But we don't have access to varmap here directly?
        # We do! 'varmap' is in the closure of generate_scope.
        return varmap.get(node["id"], "None")

    def gen_graph_output(node, inputs):
        # This node takes an input and passes it to the return statement.
        # We just return the name of the input variable.
        return inputs[0][1] if inputs else "None"

    # ========== NEW ADVANCED GENERATORS ==========

    # --- Advanced Matrix Operations ---
    def gen_cholesky(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Cholesky Decomposition")
        add(f"try:")
        add(f"    {name} = np.linalg.cholesky({A})")
        add(f"except np.linalg.LinAlgError:")
        add(f"    {name} = np.full_like({A}, np.nan)")
        return name

    def gen_matrix_rank(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Matrix Rank")
        add(f"{name} = np.linalg.matrix_rank({A})")
        return name

    def gen_matrix_norm(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        ord_prop = node.get("properties", {}).get("ord", "fro")
        add(f"# Matrix Norm")
        add(f"{name} = np.linalg.norm({A}, ord='{ord_prop}')")
        return name

    def gen_condition_number(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Condition Number")
        add(f"{name} = np.linalg.cond({A})")
        return name

    def gen_pinv(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Pseudo-Inverse (Moore-Penrose)")
        add(f"{name} = np.linalg.pinv({A})")
        return name

    def gen_null_space(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Null Space")
        add(f"_u, _s, _vh = np.linalg.svd({A})")
        add(f"_tol = max({A}.shape) * np.max(_s) * np.finfo(float).eps")
        add(f"_rank = np.sum(_s > _tol)")
        add(f"{name} = _vh[_rank:].T.conj()")
        return name

    def gen_matrix_exp(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Matrix Exponential")
        add(f"{name} = scipy.linalg.expm({A})")
        return name

    def gen_schur(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Schur Decomposition")
        add(f"{name}_T, {name}_Z = scipy.linalg.schur({A})")
        return {0: f"{name}_T", 1: f"{name}_Z"}

    def gen_hessenberg(node, inputs):
        A = inputs[0][1] if inputs else "np.eye(2)"
        name = default_var_name(node)
        add(f"# Hessenberg Form")
        add(f"{name}_H, {name}_Q = scipy.linalg.hessenberg({A}, calc_q=True)")
        return {0: f"{name}_H", 1: f"{name}_Q"}

    # --- Symbolic Math (SymPy) ---
    def gen_symbolic_simplify(node, inputs):
        name = default_var_name(node)
        expr = node.get("properties", {}).get("expression", "x**2 + 2*x + 1")
        add(f"# Symbolic Simplify")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    {name} = str(sp.simplify(sp.sympify('{expr}')))")
        add(f"except: {name} = 'Error'")
        return name

    def gen_symbolic_expand(node, inputs):
        name = default_var_name(node)
        expr = node.get("properties", {}).get("expression", "(x+1)**2")
        add(f"# Symbolic Expand")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    {name} = str(sp.expand(sp.sympify('{expr}')))")
        add(f"except: {name} = 'Error'")
        return name

    def gen_symbolic_factor(node, inputs):
        name = default_var_name(node)
        expr = node.get("properties", {}).get("expression", "x**2 - 1")
        add(f"# Symbolic Factor")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    {name} = str(sp.factor(sp.sympify('{expr}')))")
        add(f"except: {name} = 'Error'")
        return name

    def gen_symbolic_diff(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        expr = props.get("expression", "x**3 + 2*x")
        var = props.get("variable", "x")
        order = int(props.get("order", 1))
        add(f"# Symbolic Derivative")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    _x = sp.Symbol('{var}')")
        add(f"    {name} = str(sp.diff(sp.sympify('{expr}'), _x, {order}))")
        add(f"except: {name} = 'Error'")
        return name

    def gen_symbolic_integrate(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        expr = props.get("expression", "x**2")
        var = props.get("variable", "x")
        definite = props.get("definite", False)
        lower = props.get("lower", 0)
        upper = props.get("upper", 1)
        add(f"# Symbolic Integral")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    _x = sp.Symbol('{var}')")
        if definite:
            add(f"    {name} = str(sp.integrate(sp.sympify('{expr}'), (_x, {lower}, {upper})))")
        else:
            add(f"    {name} = str(sp.integrate(sp.sympify('{expr}'), _x))")
        add(f"except: {name} = 'Error'")
        return name

    def gen_symbolic_solve_eq(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        equation = props.get("equation", "x**2 - 4 = 0")
        var = props.get("variable", "x")
        add(f"# Solve Equation")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    _x = sp.Symbol('{var}')")
        add(f"    _eq = '{equation}'.replace('=', '-(')+')' if '=' in '{equation}' else '{equation}'")
        add(f"    _sols = sp.solve(sp.sympify(_eq), _x)")
        add(f"    {name} = np.array([complex(s) for s in _sols])")
        add(f"except: {name} = np.array([])")
        return name

    def gen_symbolic_solve_system(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        equations = props.get("equations", "x + y = 10; 2*x - y = 5")
        variables = props.get("variables", "x, y")
        add(f"# Solve System of Equations")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    _vars = sp.symbols('{variables}')")
        add(f"    _eqs = []")
        add(f"    for _eq_str in '{equations}'.split(';'):")
        add(f"        _eq_str = _eq_str.strip()")
        add(f"        if '=' in _eq_str:")
        add(f"            _lhs, _rhs = _eq_str.split('=')")
        add(f"            _eqs.append(sp.Eq(sp.sympify(_lhs), sp.sympify(_rhs)))")
        add(f"    {name} = sp.solve(_eqs, _vars)")
        add(f"except: {name} = {{}}")
        return name

    def gen_symbolic_limit(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        expr = props.get("expression", "sin(x)/x")
        var = props.get("variable", "x")
        point = props.get("point", "0")
        add(f"# Symbolic Limit")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    _x = sp.Symbol('{var}')")
        add(f"    {name} = str(sp.limit(sp.sympify('{expr}'), _x, sp.sympify('{point}')))")
        add(f"except: {name} = 'Error'")
        return name

    def gen_symbolic_series(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        expr = props.get("expression", "exp(x)")
        var = props.get("variable", "x")
        point = props.get("point", 0)
        order = int(props.get("order", 5))
        add(f"# Taylor Series")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    _x = sp.Symbol('{var}')")
        add(f"    {name} = str(sp.series(sp.sympify('{expr}'), _x, {point}, {order}).removeO())")
        add(f"except: {name} = 'Error'")
        return name

    def gen_symbolic_laplace(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        expr = props.get("expression", "exp(-a*t)")
        t_var = props.get("t_var", "t")
        s_var = props.get("s_var", "s")
        add(f"# Laplace Transform")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    _t, _s = sp.symbols('{t_var} {s_var}')")
        add(f"    {name} = str(sp.laplace_transform(sp.sympify('{expr}'), _t, _s, noconds=True))")
        add(f"except: {name} = 'Error'")
        return name

    def gen_symbolic_inv_laplace(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        expr = props.get("expression", "1/(s+a)")
        s_var = props.get("s_var", "s")
        t_var = props.get("t_var", "t")
        add(f"# Inverse Laplace Transform")
        add(f"try:")
        add(f"    import sympy as sp")
        add(f"    _s, _t = sp.symbols('{s_var} {t_var}')")
        add(f"    {name} = str(sp.inverse_laplace_transform(sp.sympify('{expr}'), _s, _t))")
        add(f"except: {name} = 'Error'")
        return name

    # --- Advanced Optimization ---
    def gen_assignment_problem(node, inputs):
        cost = inputs[0][1] if inputs else "np.eye(3)"
        name = default_var_name(node)
        add(f"# Assignment Problem (Hungarian Algorithm)")
        add(f"from scipy.optimize import linear_sum_assignment")
        add(f"_row_ind, _col_ind = linear_sum_assignment({cost})")
        add(f"{name}_assignment = np.array(list(zip(_row_ind, _col_ind)))")
        add(f"{name}_cost = {cost}[_row_ind, _col_ind].sum()")
        return {0: f"{name}_assignment", 1: f"{name}_cost"}

    def gen_transportation(node, inputs):
        costs = inputs[0][1] if len(inputs) > 0 else "np.ones((3,3))"
        supply = inputs[1][1] if len(inputs) > 1 else "np.array([10,10,10])"
        demand = inputs[2][1] if len(inputs) > 2 else "np.array([10,10,10])"
        name = default_var_name(node)
        add(f"# Transportation Problem")
        add(f"_costs = np.asarray({costs})")
        add(f"_supply = np.asarray({supply}).ravel()")
        add(f"_demand = np.asarray({demand}).ravel()")
        add(f"_m, _n = _costs.shape")
        add(f"_c = _costs.ravel()")
        add(f"_A_eq = []")
        add(f"_b_eq = []")
        add(f"for i in range(_m):")
        add(f"    row = np.zeros(_m * _n)")
        add(f"    row[i*_n:(i+1)*_n] = 1")
        add(f"    _A_eq.append(row)")
        add(f"    _b_eq.append(_supply[i])")
        add(f"for j in range(_n):")
        add(f"    row = np.zeros(_m * _n)")
        add(f"    for i in range(_m):")
        add(f"        row[i*_n + j] = 1")
        add(f"    _A_eq.append(row)")
        add(f"    _b_eq.append(_demand[j])")
        add(f"_res = scipy.optimize.linprog(_c, A_eq=np.array(_A_eq), b_eq=np.array(_b_eq), bounds=(0, None), method='highs')")
        add(f"{name}_flow = _res.x.reshape(_m, _n) if _res.success else np.zeros((_m, _n))")
        add(f"{name}_cost = _res.fun if _res.success else 0")
        return {0: f"{name}_flow", 1: f"{name}_cost"}

    # --- Probability Distributions ---
    def gen_normal_dist(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        mean = float(props.get("mean", 0))
        std = float(props.get("std", 1))
        n_samples = int(props.get("n_samples", 1000))
        x_range = props.get("x_range", "-4,4")
        add(f"# Normal Distribution")
        add(f"{name}_samples = np.random.normal({mean}, {std}, {n_samples})")
        add(f"_x_range = [float(x) for x in '{x_range}'.split(',')]")
        add(f"_x = np.linspace(_x_range[0]*{std}+{mean}, _x_range[1]*{std}+{mean}, 100)")
        add(f"{name}_pdf = scipy.stats.norm.pdf(_x, {mean}, {std})")
        add(f"{name}_cdf = scipy.stats.norm.cdf(_x, {mean}, {std})")
        return {0: f"{name}_samples", 1: f"{name}_pdf", 2: f"{name}_cdf"}

    def gen_uniform_dist(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        low = float(props.get("low", 0))
        high = float(props.get("high", 1))
        n_samples = int(props.get("n_samples", 1000))
        add(f"# Uniform Distribution")
        add(f"{name} = np.random.uniform({low}, {high}, {n_samples})")
        return name

    def gen_exponential_dist(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        scale = float(props.get("scale", 1.0))
        n_samples = int(props.get("n_samples", 1000))
        add(f"# Exponential Distribution")
        add(f"{name} = np.random.exponential({scale}, {n_samples})")
        return name

    def gen_poisson_dist(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        lam = float(props.get("lam", 5))
        n_samples = int(props.get("n_samples", 1000))
        add(f"# Poisson Distribution")
        add(f"{name} = np.random.poisson({lam}, {n_samples})")
        return name

    def gen_binomial_dist(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        n = int(props.get("n", 10))
        p = float(props.get("p", 0.5))
        n_samples = int(props.get("n_samples", 1000))
        add(f"# Binomial Distribution")
        add(f"{name} = np.random.binomial({n}, {p}, {n_samples})")
        return name

    # --- Combinatorics ---
    def gen_factorial(node, inputs):
        name = default_var_name(node)
        n = int(node.get("properties", {}).get("n", 5))
        add(f"# Factorial")
        add(f"import math")
        add(f"{name} = math.factorial({n})")
        return name

    def gen_binomial_coeff(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        n = int(props.get("n", 10))
        k = int(props.get("k", 3))
        add(f"# Binomial Coefficient")
        add(f"import math")
        add(f"{name} = math.comb({n}, {k})")
        return name

    def gen_permutations(node, inputs):
        elements = inputs[0][1] if inputs else "np.array([1,2,3])"
        name = default_var_name(node)
        r = node.get("properties", {}).get("r", 0)
        add(f"# Permutations")
        add(f"from itertools import permutations as _perms")
        add(f"_elems = list(np.asarray({elements}).ravel())")
        add(f"_r = {r} if {r} > 0 else len(_elems)")
        add(f"_perms_list = list(_perms(_elems, _r))")
        add(f"{name}_perms = np.array(_perms_list) if _perms_list else np.array([])")
        add(f"{name}_count = len(_perms_list)")
        return {0: f"{name}_perms", 1: f"{name}_count"}

    def gen_combinations(node, inputs):
        elements = inputs[0][1] if inputs else "np.array([1,2,3])"
        name = default_var_name(node)
        r = int(node.get("properties", {}).get("r", 2))
        add(f"# Combinations")
        add(f"from itertools import combinations as _combs")
        add(f"_elems = list(np.asarray({elements}).ravel())")
        add(f"_combs_list = list(_combs(_elems, {r}))")
        add(f"{name}_combs = np.array(_combs_list) if _combs_list else np.array([])")
        add(f"{name}_count = len(_combs_list)")
        return {0: f"{name}_combs", 1: f"{name}_count"}

    # --- Set Operations ---
    def gen_set_union(node, inputs):
        A = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        B = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add(f"# Set Union")
        add(f"{name} = np.union1d({A}, {B})")
        return name

    def gen_set_intersection(node, inputs):
        A = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        B = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add(f"# Set Intersection")
        add(f"{name} = np.intersect1d({A}, {B})")
        return name

    def gen_set_difference(node, inputs):
        A = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        B = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add(f"# Set Difference")
        add(f"{name} = np.setdiff1d({A}, {B})")
        return name

    def gen_set_symmetric_diff(node, inputs):
        A = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        B = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add(f"# Symmetric Difference")
        add(f"{name} = np.setxor1d({A}, {B})")
        return name

    def gen_cartesian_product(node, inputs):
        A = inputs[0][1] if len(inputs) > 0 else "np.array([])"
        B = inputs[1][1] if len(inputs) > 1 else "np.array([])"
        name = default_var_name(node)
        add(f"# Cartesian Product")
        add(f"from itertools import product as _prod")
        add(f"{name} = np.array(list(_prod(np.asarray({A}).ravel(), np.asarray({B}).ravel())))")
        return name

    # --- Financial Mathematics ---
    def gen_npv(node, inputs):
        cash_flows = inputs[0][1] if inputs else "np.array([-100, 30, 40, 50, 60])"
        name = default_var_name(node)
        rate = float(node.get("properties", {}).get("rate", 0.1))
        add(f"# Net Present Value")
        add(f"_cf = np.asarray({cash_flows}).ravel()")
        add(f"{name} = np.npv({rate}, _cf) if hasattr(np, 'npv') else sum(_cf[i] / (1 + {rate})**i for i in range(len(_cf)))")
        return name

    def gen_irr(node, inputs):
        cash_flows = inputs[0][1] if inputs else "np.array([-100, 30, 40, 50, 60])"
        name = default_var_name(node)
        add(f"# Internal Rate of Return")
        add(f"_cf = np.asarray({cash_flows}).ravel()")
        add(f"try:")
        add(f"    {name} = np.irr(_cf) if hasattr(np, 'irr') else scipy.optimize.brentq(lambda r: sum(_cf[i]/(1+r)**i for i in range(len(_cf))), -0.99, 10)")
        add(f"except: {name} = np.nan")
        return name

    # --- Queueing Theory ---
    def gen_mm1_queue(node, inputs):
        name = default_var_name(node)
        props = node.get("properties", {})
        lam = float(props.get("arrival_rate", 5))
        mu = float(props.get("service_rate", 8))
        add(f"# M/M/1 Queue")
        add(f"_lam, _mu = {lam}, {mu}")
        add(f"_rho = _lam / _mu")
        add(f"{name}_util = _rho")
        add(f"{name}_Lq = _rho**2 / (1 - _rho) if _rho < 1 else float('inf')")
        add(f"{name}_Wq = _lam / (_mu * (_mu - _lam)) if _rho < 1 else float('inf')")
        add(f"{name}_L = _lam / (_mu - _lam) if _rho < 1 else float('inf')")
        return {0: f"{name}_util", 1: f"{name}_Lq", 2: f"{name}_Wq", 3: f"{name}_L"}

    # --- Advanced Visualization ---
    def gen_contour_plot(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        Y = inputs[1][1] if len(inputs) > 1 else "None"
        Z = inputs[2][1] if len(inputs) > 2 else "None"
        name = default_var_name(node)
        props = node.get("properties", {})
        levels = int(props.get("levels", 10))
        title = props.get("title", "Contour Plot")
        filled = props.get("filled", True)
        add(f"# Contour Plot")
        add(f"plt.figure()")
        add(f"if {X} is not None and {Y} is not None and {Z} is not None:")
        if filled:
            add(f"    plt.contourf({X}, {Y}, {Z}, levels={levels}, cmap='viridis')")
        else:
            add(f"    plt.contour({X}, {Y}, {Z}, levels={levels})")
        add(f"    plt.colorbar()")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_quiver_plot(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        Y = inputs[1][1] if len(inputs) > 1 else "None"
        U = inputs[2][1] if len(inputs) > 2 else "None"
        V = inputs[3][1] if len(inputs) > 3 else "None"
        name = default_var_name(node)
        props = node.get("properties", {})
        title = props.get("title", "Vector Field")
        scale = float(props.get("scale", 1.0))
        add(f"# Vector Field (Quiver) Plot")
        add(f"plt.figure()")
        add(f"if {X} is not None and {Y} is not None and {U} is not None and {V} is not None:")
        add(f"    plt.quiver({X}, {Y}, {U}, {V}, scale={scale})")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_bar_plot(node, inputs):
        X = inputs[0][1] if len(inputs) > 0 else "None"
        heights = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        props = node.get("properties", {})
        title = props.get("title", "Bar Chart")
        horizontal = props.get("horizontal", False)
        add(f"# Bar Chart")
        add(f"plt.figure()")
        add(f"if {X} is not None and {heights} is not None:")
        if horizontal:
            add(f"    plt.barh({X}, {heights})")
        else:
            add(f"    plt.bar({X}, {heights})")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    def gen_pie_plot(node, inputs):
        values = inputs[0][1] if len(inputs) > 0 else "None"
        labels = inputs[1][1] if len(inputs) > 1 else "None"
        name = default_var_name(node)
        title = node.get("properties", {}).get("title", "Pie Chart")
        add(f"# Pie Chart")
        add(f"plt.figure()")
        add(f"if {values} is not None:")
        add(f"    _labels = {labels} if {labels} is not None else None")
        add(f"    plt.pie({values}, labels=_labels, autopct='%1.1f%%')")
        add(f"plt.title('{title}')")
        add(f"show_plot()")
        return name

    generators = {
        # Data Preprocessing
        "data/normalize": gen_normalize,
        "data/split": gen_train_test_split,
        "data/load_csv": gen_load_csv,
        "data/load_excel": gen_load_excel,
        "data/load_excel_adv": gen_load_excel_adv,
        "data/select_column": gen_select_column,
        "data/describe": gen_describe,
        "data/weighted_score": gen_weighted_score,
        "data/indicator_dict": gen_indicator_dict,
        "data/pivot_table": gen_pivot_table,
        "data/conditional_column": gen_conditional_column,

        # Visualization
        "viz/plot_line": gen_plot_line,
        "viz/plot_scatter": gen_plot_scatter,
        "viz/plot_hist": gen_plot_hist,
        "viz/plot_box": gen_plot_box,
        "viz/plot_heatmap": gen_plot_heatmap,
        "viz/plot_surface": gen_plot_surface,

        # New Algo
        "algo/ode_solver": gen_ode_solver,
        "algo/nonlinear_programming": gen_nlp,

        # New Stat
        "stat/ttest": gen_ttest,
        "stat/chisquare": gen_chisquare,

        # New Math
        "math/solve_linear": gen_solve_linear,
        "math/eigen": gen_eigen,
        "math/fft": gen_fft,
        "math/linspace": gen_linspace,
        "math/lu_decompose": gen_lu_decompose,
        "math/qr": gen_qr,
        "math/svd": gen_svd,
        "math/conv": gen_conv,

        # New Signal
        "signal/filter": gen_signal_filter,
        "signal/resample": gen_signal_resample,
        "signal/stft": gen_signal_stft,
        "signal/bandpass_filter": gen_bandpass_filter,
        "signal/xcorr": gen_xcorr,

        # Control
        "control/transfer_function": gen_transfer_function,
        "control/step_response": gen_step_response,
        "control/bode_plot": gen_bode,

        "math/constant": gen_constant,
        "data/vector": gen_vector,
        "data/matrix": gen_matrix,
        "data/filter_rows": gen_filter_rows,
        "data/group_aggregate": gen_group_aggregate,
        "data/rolling_window": gen_rolling_window,
        "data/transform_column": gen_transform_column,
        "data/merge_dataframes": gen_merge_dataframes,
        "data/time_features": gen_time_features,
        "data/create_dummy": gen_create_dummy,
        "data/map_values": gen_map_values,
        "data/explode_column": gen_explode_column,
        "data/expression": gen_expression,
        "math/add": binop("+"),
        "math/subtract": binop("-"),
        "math/multiply": binop("*"),
        "math/divide": binop("/"),
        "math/power": gen_power,
        "math/matmul": gen_matmul,
        "math/transpose": gen_transpose,
        "math/inverse": gen_inverse,
        "math/determinant": gen_determinant,
        "io/output": gen_output,
        "custom/python_script": gen_custom_python,
        "graph/subgraph": gen_subgraph,
        "graph/input": gen_graph_input,
        "graph/output": gen_graph_output,

        # Algo
        "algo/monte_carlo": gen_monte_carlo,
        "algo/interpolation": gen_interpolation,
        "algo/parameter_estimation": gen_parameter_estimation,
        "algo/linear_programming": gen_linprog,
        "algo/integer_programming": gen_integer_programming,
        "algo/quadratic_programming": gen_quadratic_programming,
        "algo/dijkstra": gen_dijkstra,
        "algo/mst": gen_mst,
        "algo/max_flow": gen_max_flow,
        "algo/dynamic_programming": gen_dynamic_programming,
        "algo/backtracking": gen_backtracking,
        "algo/divide_conquer": gen_divide_conquer,
        "algo/simulated_annealing": gen_simulated_annealing,
        "algo/genetic_algorithm": gen_genetic_algorithm,
        "algo/neural_network_opt": gen_neural_network_opt,
        "algo/grid_search": gen_grid_search,
        "algo/exhaustive_search": gen_exhaustive_search,
        "algo/discretize": gen_discretize,
        "algo/numerical_integration": gen_numerical_integration,
        "algo/root_finding": gen_root_finding,
        "algo/image_filter": gen_image_filter,

        # Models
        "model/bp_neural_network": gen_bp_nn,
        "model/polynomial_fitting": gen_poly_fit,
        "model/svm_predict": gen_svm,
        "model/grey_prediction": gen_grey_prediction,
        "model/time_series": gen_time_series,
        "model/markov_chain": gen_markov_chain,
        "model/linear_regression_fit": gen_lr_fit,
        "model/ridge_regression": gen_ridge,
        "model/lasso_regression": gen_lasso,
        "model/poly_features": gen_poly_features,
        "model/predict": gen_predict,

        # Eval
        "eval/ahp": gen_ahp,
        "eval/topsis": gen_topsis,
        "eval/fuzzy_eval": gen_fuzzy_eval,
        "eval/grey_relational": gen_grey_relational,
        "eval/pca": gen_pca,
        "eval/rsr": gen_rsr,
        "eval/coupling": gen_coupling,
        "eval/bp_eval": gen_bp_eval,
        "metrics/mse": gen_mse,
        "metrics/mae": gen_mae,
        "metrics/rmse": gen_rmse,
        "metrics/r2": gen_r2,
        "metrics/accuracy": gen_accuracy,

        # Class
        "class/kmeans": gen_kmeans,
        "class/decision_tree": gen_decision_tree,
        "class/logistic_regression": gen_logistic_regression,
        "class/random_forest": gen_random_forest,
        "class/naive_bayes": gen_naive_bayes,

        # Opt
        "opt/multiobjective_weighted": gen_multiobjective_weighted,
        "opt/linear_model_builder": gen_linear_model_builder,
        "opt/variable_builder": gen_variable_builder,
        "opt/constraint_builder": gen_constraint_builder,
        "opt/export_lp": gen_export_lp,
        "opt/solution_report": gen_solution_report,
        "opt/knapsack": gen_knapsack,
        "opt/tsp": gen_tsp,
        "opt/vrp": gen_vrp,

        # Stat
        "stat/correlation": gen_correlation,
        "stat/anova": gen_anova,
        "stat/discriminant": gen_discriminant,
        "stat/autocorr": gen_autocorr,
        "stat/pacf": gen_pacf,

        # Advanced Matrix Operations
        "math/cholesky": gen_cholesky,
        "math/matrix_rank": gen_matrix_rank,
        "math/matrix_norm": gen_matrix_norm,
        "math/condition_number": gen_condition_number,
        "math/pinv": gen_pinv,
        "math/null_space": gen_null_space,
        "math/matrix_exp": gen_matrix_exp,
        "math/schur": gen_schur,
        "math/hessenberg": gen_hessenberg,

        # Symbolic Math
        "symbolic/simplify": gen_symbolic_simplify,
        "symbolic/expand": gen_symbolic_expand,
        "symbolic/factor": gen_symbolic_factor,
        "symbolic/diff": gen_symbolic_diff,
        "symbolic/integrate": gen_symbolic_integrate,
        "symbolic/solve_eq": gen_symbolic_solve_eq,
        "symbolic/solve_system": gen_symbolic_solve_system,
        "symbolic/limit": gen_symbolic_limit,
        "symbolic/series": gen_symbolic_series,
        "symbolic/laplace": gen_symbolic_laplace,
        "symbolic/inv_laplace": gen_symbolic_inv_laplace,

        # Advanced Optimization
        "opt/assignment_problem": gen_assignment_problem,
        "opt/transportation": gen_transportation,

        # Probability Distributions
        "prob/normal_dist": gen_normal_dist,
        "prob/uniform_dist": gen_uniform_dist,
        "prob/exponential_dist": gen_exponential_dist,
        "prob/poisson_dist": gen_poisson_dist,
        "prob/binomial_dist": gen_binomial_dist,

        # Combinatorics
        "comb/factorial": gen_factorial,
        "comb/binomial": gen_binomial_coeff,
        "comb/permutations": gen_permutations,
        "comb/combinations": gen_combinations,

        # Set Operations
        "set/union": gen_set_union,
        "set/intersection": gen_set_intersection,
        "set/difference": gen_set_difference,
        "set/symmetric_diff": gen_set_symmetric_diff,
        "set/cartesian_product": gen_cartesian_product,

        # Financial Mathematics
        "finance/npv": gen_npv,
        "finance/irr": gen_irr,

        # Queueing Theory
        "queue/mm1": gen_mm1_queue,

        # Advanced Visualization
        "viz/contour": gen_contour_plot,
        "viz/quiver": gen_quiver_plot,
        "viz/bar": gen_bar_plot,
        "viz/pie": gen_pie_plot,
    }

    # Loop over nodes in topological order
    for nid in order:
        node = g.nodes[nid]
        ntype = node.get("type")
        inputs = resolve_input_vars(node, g, varmap)
        gen = generators.get(ntype)

        # Add section comment when category changes
        add_section_comment(ntype)

        if gen is None:
            vname = default_var_name(node)
            add(f"# Node type '{ntype}' not recognized; placeholder created")
            add(f"{vname} = None")
            varmap[nid] = vname
            continue

        # Add descriptive comment for this node
        add_node_comment(node)

        vname = gen(node, inputs)
        varmap[nid] = vname

    return lines, varmap


def analyze_required_imports(graph_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Analyze graph to determine which imports are actually needed."""
    g = Graph(graph_data)
    node_types = set()
    for nid in g.nodes:
        node = g.nodes[nid]
        ntype = node.get("type", "")
        node_types.add(ntype)

    # Define import groups based on node types
    import_groups = {
        "numpy": {
            "imports": ["import numpy as np"],
            "triggers": ["math/", "data/vector", "data/matrix", "algo/", "signal/", "eval/"]
        },
        "pandas": {
            "imports": ["import pandas as pd"],
            "triggers": ["data/load_csv", "data/load_excel", "data/filter_rows", "data/group_aggregate",
                        "data/pivot_table", "data/merge_dataframes", "data/describe", "data/select_column",
                        "data/rolling_window", "data/transform_column", "data/time_features", "data/create_dummy",
                        "data/map_values", "data/explode_column", "data/expression", "data/conditional_column"]
        },
        "scipy_optimize": {
            "imports": ["import scipy.optimize"],
            "triggers": ["algo/linear_programming", "algo/integer_programming", "algo/quadratic_programming",
                        "algo/nonlinear_programming", "algo/parameter_estimation"]
        },
        "scipy_interpolate": {
            "imports": ["import scipy.interpolate"],
            "triggers": ["algo/interpolation"]
        },
        "scipy_integrate": {
            "imports": ["import scipy.integrate"],
            "triggers": ["algo/ode_solver", "algo/numerical_integration"]
        },
        "scipy_linalg": {
            "imports": ["import scipy.linalg"],
            "triggers": ["math/lu_decompose", "math/qr", "math/svd", "math/cholesky"]
        },
        "scipy_stats": {
            "imports": ["import scipy.stats"],
            "triggers": ["stat/ttest", "stat/chisquare", "stat/anova", "stat/correlation"]
        },
        "scipy_signal": {
            "imports": ["import scipy.signal"],
            "triggers": ["signal/filter", "signal/resample", "signal/stft", "signal/bandpass_filter",
                        "signal/xcorr", "math/fft", "math/conv"]
        },
        "networkx": {
            "imports": ["import networkx as nx"],
            "triggers": ["algo/dijkstra", "algo/mst", "algo/max_flow"]
        },
        "sklearn_nn": {
            "imports": ["from sklearn.neural_network import MLPRegressor, MLPClassifier"],
            "triggers": ["model/mlp_regression", "class/mlp_classifier"]
        },
        "sklearn_svm": {
            "imports": ["from sklearn.svm import SVR, SVC"],
            "triggers": ["model/svr", "class/svc"]
        },
        "sklearn_cluster": {
            "imports": ["from sklearn.cluster import KMeans"],
            "triggers": ["class/kmeans"]
        },
        "sklearn_tree": {
            "imports": ["from sklearn.tree import DecisionTreeClassifier"],
            "triggers": ["class/decision_tree"]
        },
        "sklearn_ensemble": {
            "imports": ["from sklearn.ensemble import RandomForestClassifier"],
            "triggers": ["class/random_forest"]
        },
        "sklearn_linear": {
            "imports": ["from sklearn.linear_model import LogisticRegression",
                       "from sklearn.linear_model import Ridge, Lasso"],
            "triggers": ["class/logistic_regression", "model/ridge_regression", "model/lasso_regression"]
        },
        "sklearn_naive_bayes": {
            "imports": ["from sklearn.naive_bayes import GaussianNB"],
            "triggers": ["class/naive_bayes"]
        },
        "sklearn_decomposition": {
            "imports": ["from sklearn.decomposition import PCA"],
            "triggers": ["eval/pca"]
        },
        "sklearn_discriminant": {
            "imports": ["from sklearn.discriminant_analysis import LinearDiscriminantAnalysis"],
            "triggers": ["eval/lda"]
        },
        "sklearn_preprocessing": {
            "imports": ["from sklearn.preprocessing import PolynomialFeatures"],
            "triggers": ["model/polynomial_regression"]
        },
        "statsmodels": {
            "imports": ["import statsmodels.api as sm"],
            "triggers": ["model/linear_regression_fit", "stat/"]
        },
        "matplotlib": {
            "imports": [
                "import matplotlib.pyplot as plt",
                "import matplotlib",
                "matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']",
                "matplotlib.rcParams['axes.unicode_minus'] = False"
            ],
            "triggers": ["viz/"]
        },
        "matplotlib_3d": {
            "imports": ["from mpl_toolkits.mplot3d import Axes3D"],
            "triggers": ["viz/plot_surface"]
        },
        "io_base64": {
            "imports": ["import io", "import base64"],
            "triggers": ["viz/"]
        },
        "sympy": {
            "imports": ["import sympy as sp", "from sympy import symbols, simplify, expand, factor, diff, integrate, solve, limit, series"],
            "triggers": ["symbolic/"]
        },
        "control": {
            "imports": ["import control"],
            "triggers": ["control/"]
        }
    }

    required_imports = []
    for group_name, group_info in import_groups.items():
        for trigger in group_info["triggers"]:
            for ntype in node_types:
                if ntype.startswith(trigger) or ntype == trigger.rstrip("/"):
                    required_imports.extend(group_info["imports"])
                    break
            else:
                continue
            break

    # Remove duplicates while preserving order
    seen = set()
    unique_imports = []
    for imp in required_imports:
        if imp not in seen:
            seen.add(imp)
            unique_imports.append(imp)

    return unique_imports


def get_node_category(ntype: str) -> str:
    """Get human-readable category name for a node type."""
    categories = {
        "math/": "数学运算 (Mathematical Operations)",
        "data/": "数据处理 (Data Processing)",
        "algo/": "算法 (Algorithms)",
        "viz/": "可视化 (Visualization)",
        "model/": "模型训练 (Model Training)",
        "class/": "分类算法 (Classification)",
        "eval/": "评估方法 (Evaluation Methods)",
        "stat/": "统计分析 (Statistical Analysis)",
        "signal/": "信号处理 (Signal Processing)",
        "symbolic/": "符号计算 (Symbolic Computation)",
        "control/": "控制系统 (Control Systems)",
        "io/": "输入输出 (Input/Output)",
        "graph/": "子图 (Subgraph)",
        "custom/": "自定义 (Custom)",
        "queue/": "排队论 (Queueing Theory)"
    }
    for prefix, category in categories.items():
        if ntype.startswith(prefix):
            return category
    return "其他 (Other)"


def get_node_description(ntype: str) -> str:
    """Get human-readable description for a node type."""
    descriptions = {
        # Math
        "math/constant": "定义常量",
        "math/add": "加法运算",
        "math/subtract": "减法运算",
        "math/multiply": "乘法运算",
        "math/divide": "除法运算",
        "math/power": "幂运算",
        "math/matmul": "矩阵乘法",
        "math/transpose": "矩阵转置",
        "math/inverse": "矩阵求逆",
        "math/determinant": "矩阵行列式",
        "math/eigen": "特征值分解",
        "math/lu_decompose": "LU分解",
        "math/qr": "QR分解",
        "math/svd": "奇异值分解",
        "math/fft": "快速傅里叶变换",
        "math/linspace": "等差数列生成",
        "math/solve_linear": "求解线性方程组",
        "math/conv": "卷积运算",
        # Data
        "data/vector": "定义向量",
        "data/matrix": "定义矩阵",
        "data/load_csv": "加载CSV文件",
        "data/load_excel": "加载Excel文件",
        "data/load_excel_adv": "加载Excel文件(高级)",
        "data/normalize": "数据标准化",
        "data/split": "数据集划分",
        "data/filter_rows": "行过滤",
        "data/group_aggregate": "分组聚合",
        "data/pivot_table": "数据透视表",
        "data/select_column": "选择列",
        "data/describe": "数据描述统计",
        "data/merge_dataframes": "合并数据框",
        "data/rolling_window": "滑动窗口",
        "data/transform_column": "列变换",
        "data/time_features": "时间特征提取",
        "data/create_dummy": "创建哑变量",
        "data/map_values": "值映射",
        "data/explode_column": "列展开",
        "data/expression": "表达式计算",
        "data/conditional_column": "条件列",
        "data/weighted_score": "加权评分",
        "data/indicator_dict": "指标字典",
        # Algo
        "algo/linear_programming": "线性规划",
        "algo/integer_programming": "整数规划",
        "algo/quadratic_programming": "二次规划",
        "algo/nonlinear_programming": "非线性规划",
        "algo/dijkstra": "Dijkstra最短路径",
        "algo/mst": "最小生成树",
        "algo/max_flow": "最大流",
        "algo/monte_carlo": "蒙特卡洛模拟",
        "algo/interpolation": "插值",
        "algo/parameter_estimation": "参数估计",
        "algo/ode_solver": "常微分方程求解",
        "algo/simulated_annealing": "模拟退火",
        "algo/genetic_algorithm": "遗传算法",
        "algo/dynamic_programming": "动态规划",
        "algo/backtracking": "回溯算法",
        "algo/divide_conquer": "分治算法",
        "algo/numerical_integration": "数值积分",
        # Viz
        "viz/plot_line": "折线图",
        "viz/plot_scatter": "散点图",
        "viz/plot_hist": "直方图",
        "viz/plot_box": "箱线图",
        "viz/plot_heatmap": "热力图",
        "viz/plot_surface": "3D曲面图",
        "viz/contour": "等高线图",
        "viz/quiver": "向量场图",
        "viz/bar": "柱状图",
        "viz/pie": "饼图",
        # Model
        "model/linear_regression_fit": "线性回归",
        "model/ridge_regression": "岭回归",
        "model/lasso_regression": "Lasso回归",
        "model/polynomial_regression": "多项式回归",
        "model/mlp_regression": "MLP回归",
        "model/svr": "支持向量回归",
        # Class
        "class/kmeans": "K-Means聚类",
        "class/decision_tree": "决策树",
        "class/logistic_regression": "逻辑回归",
        "class/random_forest": "随机森林",
        "class/naive_bayes": "朴素贝叶斯",
        "class/svc": "支持向量分类",
        "class/mlp_classifier": "MLP分类器",
        # Eval
        "eval/ahp": "层次分析法(AHP)",
        "eval/topsis": "TOPSIS评价",
        "eval/pca": "主成分分析(PCA)",
        "eval/grey_relational": "灰色关联分析",
        "eval/lda": "线性判别分析(LDA)",
        # Stat
        "stat/ttest": "t检验",
        "stat/chisquare": "卡方检验",
        "stat/anova": "方差分析",
        "stat/correlation": "相关性分析",
        # Signal
        "signal/filter": "信号滤波",
        "signal/resample": "信号重采样",
        "signal/stft": "短时傅里叶变换",
        "signal/bandpass_filter": "带通滤波",
        "signal/xcorr": "互相关",
        # Control
        "control/transfer_function": "传递函数",
        "control/step_response": "阶跃响应",
        "control/bode_plot": "Bode图",
        # IO
        "io/output": "输出结果",
        # Graph
        "graph/subgraph": "子图调用",
        "graph/input": "子图输入",
        "graph/output": "子图输出",
        # Custom
        "custom/python_script": "自定义Python脚本",
        # Queue
        "queue/mm1": "M/M/1排队模型"
    }
    return descriptions.get(ntype, ntype.split("/")[-1])


def generate_code(graph_data: Dict[str, Any], readable: bool = True) -> str:
    """
    Generate Python code from graph data.

    Args:
        graph_data: The graph structure containing nodes and links
        readable: If True, generate more readable code with comments and sections
    """
    g = Graph(graph_data)

    # Analyze required imports
    required_imports = analyze_required_imports(graph_data)

    # Always need numpy for basic operations
    if "import numpy as np" not in required_imports:
        required_imports.insert(0, "import numpy as np")

    # Build header
    header_lines = [
        "# " + "=" * 70,
        "# AlgoNode 生成的代码 (Generated by AlgoNode)",
        "# " + "=" * 70,
        "#",
        "# 运行环境要求 (Requirements):",
        "#   pip install numpy scipy scikit-learn networkx statsmodels pandas matplotlib",
        "#",
        "# " + "=" * 70,
        ""
    ]

    # Build imports section
    import_lines = [
        "# " + "-" * 40,
        "# 导入依赖库 (Import Dependencies)",
        "# " + "-" * 40,
    ]
    import_lines.extend(required_imports)
    import_lines.append("")

    # Check if visualization is needed
    has_viz = any(node.get("type", "").startswith("viz/") for node in g.nodes.values())

    # Build helper functions section
    helper_lines = []
    if has_viz:
        helper_lines = [
            "# " + "-" * 40,
            "# 辅助函数 (Helper Functions)",
            "# " + "-" * 40,
            "",
            "def show_plot():",
            "    \"\"\"将图表转换为Base64编码的图片并输出\"\"\"",
            "    buf = io.BytesIO()",
            "    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')",
            "    buf.seek(0)",
            "    img_base64 = base64.b64encode(buf.read()).decode('utf-8')",
            "    print(f'<img src=\"data:image/png;base64,{img_base64}\" />')",
            "    plt.close()",
            "",
        ]

    # Generate main code with sections
    main_lines, main_varmap = generate_scope(graph_data, readable=readable)

    # Collect outputs from main scope
    output_vars = []
    for nid in g.topo_order():
        node = g.nodes[nid]
        if node.get("type") == "io/output":
            val = main_varmap.get(nid)
            if val:
                output_vars.append(val)

    # Build main code section
    code_section = [
        "# " + "-" * 40,
        "# 主程序 (Main Program)",
        "# " + "-" * 40,
        ""
    ]
    code_section.extend(main_lines)

    # Build output section
    output_lines = []
    if output_vars:
        output_lines = [
            "",
            "# " + "-" * 40,
            "# 输出结果 (Output Results)",
            "# " + "-" * 40,
        ]
        for v in output_vars:
            output_lines.append(f"print('{v} =', {v})")

    # Combine all sections
    all_lines = []
    all_lines.extend(header_lines)
    all_lines.extend(import_lines)
    all_lines.extend(helper_lines)
    all_lines.extend(code_section)
    all_lines.extend(output_lines)

    return "\n".join(all_lines)


# -------------------------
# Flask Routes
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# User Library Management
# -------------------------
import os

USER_LIBRARY_FILE = os.path.join(os.path.dirname(__file__), "user_library.json")

def load_user_library():
    """Load user's custom node library"""
    if os.path.exists(USER_LIBRARY_FILE):
        try:
            with open(USER_LIBRARY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"nodes": []}
    return {"nodes": []}

def save_user_library(library):
    """Save user's custom node library"""
    with open(USER_LIBRARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(library, f, ensure_ascii=False, indent=2)


@app.route("/library", methods=["GET"])
def get_library():
    """Get all user library nodes"""
    library = load_user_library()
    return jsonify({"ok": True, "library": library})


@app.route("/library/add", methods=["POST"])
def add_to_library():
    """Add a node (subgraph or custom script) to user library"""
    try:
        payload = request.get_json(force=True)
        node_data = payload.get("node")
        name = payload.get("name", "").strip()
        description = payload.get("description", "").strip()
        category = payload.get("category", "我的节点").strip()
        
        if not node_data:
            return jsonify({"ok": False, "error": "No node data provided"}), 400
        
        if not name:
            name = node_data.get("title") or f"Custom_{node_data.get('id', 0)}"
        
        library = load_user_library()
        
        # Generate unique ID
        max_id = max([n.get("lib_id", 0) for n in library.get("nodes", [])] + [0])
        new_id = max_id + 1
        
        library_entry = {
            "lib_id": new_id,
            "name": name,
            "description": description,
            "category": category,
            "node_type": node_data.get("type"),
            "node_data": node_data,
            "created_at": __import__('datetime').datetime.now().isoformat()
        }
        
        library.setdefault("nodes", []).append(library_entry)
        save_user_library(library)
        
        return jsonify({"ok": True, "lib_id": new_id, "message": f"已添加 '{name}' 到库中"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/library/delete/<int:lib_id>", methods=["DELETE"])
def delete_from_library(lib_id):
    """Delete a node from user library"""
    try:
        library = load_user_library()
        library["nodes"] = [n for n in library.get("nodes", []) if n.get("lib_id") != lib_id]
        save_user_library(library)
        return jsonify({"ok": True, "message": "已从库中删除"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/library/update/<int:lib_id>", methods=["PUT"])
def update_library_node(lib_id):
    """Update a library node's metadata"""
    try:
        payload = request.get_json(force=True)
        library = load_user_library()
        
        for node in library.get("nodes", []):
            if node.get("lib_id") == lib_id:
                if "name" in payload:
                    node["name"] = payload["name"]
                if "description" in payload:
                    node["description"] = payload["description"]
                if "category" in payload:
                    node["category"] = payload["category"]
                break
        
        save_user_library(library)
        return jsonify({"ok": True, "message": "已更新"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# -------------------------
# Data Management
# -------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "uploads")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route("/api/data/list", methods=["GET"])
def list_data_files():
    files = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, f)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                files.append({
                    "name": f,
                    "size": size,
                    "path": path.replace("\\", "/") # Normalize for JS
                })
    return jsonify({"ok": True, "files": files})

@app.route("/api/data/upload", methods=["POST"])
def upload_data_file():
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No selected file"}), 400
    if file:
        filename = file.filename
        # Basic security: prevent directory traversal
        filename = os.path.basename(filename)
        file.save(os.path.join(DATA_DIR, filename))
        return jsonify({"ok": True, "message": "File uploaded successfully"})
    return jsonify({"ok": False, "error": "Upload failed"}), 400

@app.route("/api/data/delete/<filename>", methods=["DELETE"])
def delete_data_file(filename):
    filename = os.path.basename(filename)
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            os.remove(path)
            return jsonify({"ok": True, "message": "Deleted"})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": False, "error": "File not found"}), 404


@app.route("/export", methods=["POST"])
def export_code():
    try:
        payload = request.get_json(force=True)
        graph = payload.get("graph")
        if isinstance(graph, str):
            graph = json.loads(graph)
        code = generate_code(graph)
        return jsonify({"ok": True, "code": code})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/run", methods=["POST"])
def run_code():
    import subprocess
    import sys
    import os
    import tempfile

    try:
        payload = request.get_json(force=True)
        graph = payload.get("graph")
        if isinstance(graph, str):
            graph = json.loads(graph)
        code = generate_code(graph)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name

        # Run
        try:
            # Use system python to execute the script
            # This assumes 'python' is in the system PATH
            python_cmd = "python"
            
            # Allow configuration via environment variable if needed
            if "PYTHON_PATH" in os.environ:
                python_cmd = os.environ["PYTHON_PATH"]

            result = subprocess.run([python_cmd, temp_path], capture_output=True, text=True, timeout=15)
            output = result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            output = "Error: Execution timed out (15s)"
        except Exception as e:
            output = f"Error executing script: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify({"ok": True, "output": output})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# -------------------------
# Node Market
# -------------------------
MARKET_DIR = os.path.join(os.path.dirname(__file__), "market")

def _local_market_list():
    """Get market items from local storage"""
    if not os.path.exists(MARKET_DIR):
        os.makedirs(MARKET_DIR)
    
    items = []
    for filename in os.listdir(MARKET_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(MARKET_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items.append({
                        "filename": filename,
                        "name": data.get("name", filename),
                        "description": data.get("description", ""),
                        "author": data.get("author", "Anonymous"),
                        "type": data.get("type", "project"),
                        "timestamp": data.get("timestamp", ""),
                        "source": "local"
                    })
            except:
                pass
    return items

def _remote_market_list():
    """Get market items from central server"""
    if not CENTRAL_SERVER_URL:
        return []
    try:
        resp = requests.get(f"{CENTRAL_SERVER_URL}/api/market/list", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                items = data.get("items", [])
                # Mark items as from remote
                for item in items:
                    item["source"] = "remote"
                return items
    except Exception as e:
        print(f"[Market] Failed to fetch from central server: {e}")
    return []

@app.route("/api/market/list", methods=["GET"])
def market_list():
    """List market items from both local and central server"""
    items = []
    
    # Always get local items
    items.extend(_local_market_list())
    
    # If central server is configured and we're in client mode, also fetch remote items
    if CENTRAL_SERVER_URL and ALGONODE_MODE != "server":
        remote_items = _remote_market_list()
        items.extend(remote_items)
    
    # Sort by timestamp descending
    items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    
    return jsonify({"ok": True, "items": items})

@app.route("/api/market/upload", methods=["POST"])
def market_upload():
    """Upload item to market (local or central server)"""
    try:
        payload = request.get_json(force=True)
        name = payload.get("name", "Untitled")
        description = payload.get("description", "")
        author = payload.get("author", "Anonymous")
        item_type = payload.get("type", "project")
        content = payload.get("content")
        target = payload.get("target", "local")  # "local" or "remote"
        
        if not content:
            return jsonify({"ok": False, "error": "No content provided"}), 400
        
        # If target is remote and central server is configured
        if target == "remote" and CENTRAL_SERVER_URL:
            try:
                resp = requests.post(
                    f"{CENTRAL_SERVER_URL}/api/market/upload",
                    json={
                        "name": name,
                        "description": description,
                        "author": author,
                        "type": item_type,
                        "content": content,
                        "target": "local"  # On server side, save locally
                    },
                    timeout=30
                )
                if resp.status_code == 200:
                    return jsonify(resp.json())
                else:
                    return jsonify({"ok": False, "error": f"Central server error: {resp.status_code}"}), 500
            except Exception as e:
                return jsonify({"ok": False, "error": f"Failed to upload to central server: {str(e)}"}), 500
        
        # Save locally
        if not os.path.exists(MARKET_DIR):
            os.makedirs(MARKET_DIR)
            
        import time
        timestamp = int(time.time())
        filename = f"{timestamp}_{sanitize_name(name)}.json"
        filepath = os.path.join(MARKET_DIR, filename)
        
        data = {
            "name": name,
            "description": description,
            "author": author,
            "type": item_type,
            "timestamp": timestamp,
            "content": content
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return jsonify({"ok": True, "message": "Upload successful"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/market/import", methods=["POST"])
def market_import():
    """Import item from market (local or remote)"""
    try:
        payload = request.get_json(force=True)
        filename = payload.get("filename")
        source = payload.get("source", "local")  # "local" or "remote"
        
        data = None
        
        if source == "remote" and CENTRAL_SERVER_URL:
            # Fetch from central server
            try:
                resp = requests.post(
                    f"{CENTRAL_SERVER_URL}/api/market/import",
                    json={"filename": filename, "source": "local"},
                    timeout=30
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("ok"):
                        # For project type, content is returned directly
                        if result.get("type") == "project":
                            return jsonify(result)
                        # For node/subgraph, we need to get full data from server
                        # The server returns the processed result, just pass it through
                        return jsonify(result)
                return jsonify({"ok": False, "error": "Failed to fetch from central server"}), 500
            except Exception as e:
                return jsonify({"ok": False, "error": f"Central server error: {str(e)}"}), 500
        
        # Local import
        filepath = os.path.join(MARKET_DIR, filename)
        if not os.path.exists(filepath):
             return jsonify({"ok": False, "error": "Item not found"}), 404
             
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        item_type = data.get("type", "project")
        content = data.get("content")
        
        if item_type in ("node", "subgraph"):
            # Add to user library
            library = load_user_library()
            max_id = max([n.get("lib_id", 0) for n in library.get("nodes", [])] + [0])
            new_id = max_id + 1
            
            node_type = content.get("type") if isinstance(content, dict) else "unknown"
            category = "市场-子图" if item_type == "subgraph" else "市场-节点"
            
            library_entry = {
                "lib_id": new_id,
                "name": data.get("name"),
                "description": data.get("description"),
                "category": category,
                "node_type": node_type,
                "node_data": content,
                "created_at": __import__('datetime').datetime.now().isoformat()
            }
            library.setdefault("nodes", []).append(library_entry)
            save_user_library(library)
            return jsonify({"ok": True, "type": item_type, "message": "Imported to User Library"})
            
        else:
            # Project: Return content to frontend to load
            return jsonify({"ok": True, "type": "project", "content": content})
            
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/market/config", methods=["GET"])
def market_config():
    """Return market configuration to frontend"""
    # Only expose central server URL to clients. If this instance is running
    # in server mode, do not advertise the central_server value (it may be set
    # to the server's own address and would otherwise show as "已连接..." on
    # the server instance). Instead return an explicit flag.
    central = None
    connected = False
    
    if CENTRAL_SERVER_URL and ALGONODE_MODE != "server":
        central = CENTRAL_SERVER_URL
        # Check connection
        try:
            # Use a short timeout to check connectivity
            resp = requests.get(f"{CENTRAL_SERVER_URL}/api/market/config", timeout=2)
            if resp.status_code == 200:
                connected = True
        except:
            connected = False

    return jsonify({
        "ok": True,
        "central_server": central,
        "connected": connected,
        "mode": ALGONODE_MODE,
        "is_central_server": ALGONODE_MODE == "server"
    })


# -------------------------
# Admin Routes
# -------------------------

@app.route("/admin")
def admin_page():
    return render_template("admin.html")

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    try:
        payload = request.get_json(force=True)
        password = payload.get("password")
        if password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return jsonify({"ok": True})
        else:
            return jsonify({"ok": False, "error": "密码错误"}), 401
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin_logged_in", None)
    return jsonify({"ok": True})

@app.route("/api/admin/check_session", methods=["GET"])
def admin_check_session():
    return jsonify({"ok": True, "logged_in": session.get("admin_logged_in", False)})

@app.route("/api/admin/market/delete", methods=["POST"])
def admin_market_delete():
    if not session.get("admin_logged_in"):
        return jsonify({"ok": False, "error": "未登录"}), 401
        
    try:
        payload = request.get_json(force=True)
        filename = payload.get("filename")
        if not filename:
            return jsonify({"ok": False, "error": "Missing filename"}), 400
            
        # Security check: ensure filename is just a name, not a path
        if os.path.basename(filename) != filename:
             return jsonify({"ok": False, "error": "Invalid filename"}), 400
             
        filepath = os.path.join(MARKET_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"ok": True, "message": "Deleted"})
        else:
            return jsonify({"ok": False, "error": "File not found"}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() in ("true", "1", "yes")
    
    print(f"[AlgoNode] Starting server on {host}:{port}")
    if CENTRAL_SERVER_URL:
        print(f"[AlgoNode] Central server: {CENTRAL_SERVER_URL}")
    print(f"[AlgoNode] Mode: {ALGONODE_MODE}")
    
    # Auto-open browser if not in server mode
    if ALGONODE_MODE != "server":
        # Avoid opening twice when debug reloader is active
        if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            import webbrowser
            from threading import Timer
            def open_browser():
                webbrowser.open(f"http://127.0.0.1:{port}")
            Timer(1.5, open_browser).start()

    app.run(host=host, port=port, debug=debug)

    # This way requires python installation at runtime.
