# AlgoNode 增强节点设计方案
## 目标：极大减少自定义 Python 节点的需求

### 当前问题分析
在 MCM Problem C 的建模中，我们不得不编写大量自定义 Python 代码来实现：
- 移动平均（3届/5届）
- 差分/动量计算
- 数据合并（奖牌+项目+主办国）
- 时间特征提取
- 条件填充

这些都是**通用的数据处理操作**，应该被封装成独立节点。

---

## 新增节点设计

### 1. **data/rolling_window** - 滚动窗口计算
**功能**：计算移动平均、移动求和等滚动统计量

**输入**：
- Data (DataFrame)

**输出**：
- Result (DataFrame，包含新列)

**参数**：
- `column`: 要处理的列名（如 "Gold"）
- `window`: 窗口大小（如 3, 5）
- `operation`: 操作类型
  - `mean` - 移动平均
  - `sum` - 移动求和
  - `std` - 移动标准差
  - `min/max` - 移动最小/最大值
- `groupby`: 分组列（如 "NOC"，按国家分组计算）
- `min_periods`: 最小有效期数（默认1）
- `output_column`: 输出列名（如 "gold_avg_3"）

**示例配置**：
```json
{
  "column": "Gold",
  "window": 3,
  "operation": "mean",
  "groupby": "NOC",
  "min_periods": 1,
  "output_column": "gold_avg_3"
}
```

**替代代码**：
```python
# 原自定义代码：
medals["gold_avg_3"] = medals.groupby("NOC")["Gold"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

# 现在只需拖一个节点即可
```

---

### 2. **data/transform_column** - 列变换
**功能**：对单列进行各种数学/统计变换

**输入**：
- Data (DataFrame)

**输出**：
- Result (DataFrame，包含新列)

**参数**：
- `column`: 源列名
- `operation`: 变换类型
  - `diff` - 差分（一阶导数）
  - `pct_change` - 百分比变化
  - `shift` - 滞后/超前
  - `cumsum` - 累计求和
  - `log` - 对数变换
  - `sqrt` - 平方根
  - `abs` - 绝对值
  - `fillna` - 填充缺失值
  - `round` - 四舍五入
- `periods`: 周期数（用于 diff/shift，默认1）
- `fill_value`: 填充值（用于 fillna，默认0）
- `decimals`: 小数位数（用于 round）
- `output_column`: 输出列名

**示例配置**：
```json
{
  "column": "Gold",
  "operation": "diff",
  "periods": 1,
  "output_column": "gold_momentum"
}
```

**替代代码**：
```python
# 原自定义代码：
medals["gold_momentum"] = medals.groupby("NOC")["Gold"].diff().fillna(0)

# 现在拖两个节点：
# 1. data/transform_column (operation=diff, groupby支持)
# 2. data/transform_column (operation=fillna, fill_value=0)
```

---

### 3. **data/merge_dataframes** - 数据合并
**功能**：合并两个 DataFrame（类似 SQL JOIN）

**输入**：
- Left (DataFrame)
- Right (DataFrame)

**输出**：
- Merged (DataFrame)

**参数**：
- `how`: 合并方式
  - `inner` - 内连接
  - `left` - 左连接
  - `right` - 右连接
  - `outer` - 全外连接
- `on`: 共同键列（如 "Year"）
- `left_on`: 左表键列
- `right_on`: 右表键列

**示例配置**：
```json
{
  "how": "left",
  "left_on": "Year",
  "right_on": "Year"
}
```

**替代代码**：
```python
# 原自定义代码：
merged = pd.merge(medals, hosts, on="Year", how="left")

# 现在拖一个 data/merge_dataframes 节点
```

---

### 4. **data/time_features** - 时间特征提取
**功能**：从日期/年份列自动生成时间特征

**输入**：
- Data (DataFrame)

**输出**：
- Features (DataFrame，包含新时间列)

**参数**：
- `date_column`: 日期列名（如 "Year"）
- `features`: 要提取的特征（逗号分隔）
  - `year` - 年份
  - `month` - 月份
  - `day` - 日期
  - `dayofweek` - 星期几
  - `quarter` - 季度
  - `dayofyear` - 一年中的第几天
  - `weekofyear` - 一年中的第几周

**示例配置**：
```json
{
  "date_column": "Year",
  "features": "year"
}
```

**替代代码**：
```python
# 原自定义代码：
base_year = medals["Year"].min()
medals["year_centered"] = medals["Year"] - base_year

# 现在用 data/time_features + data/transform_column
```

---

### 5. **data/create_dummy** - 虚拟变量生成
**功能**：创建 one-hot 编码或二值标记

**输入**：
- Data (DataFrame)

**输出**：
- Result (DataFrame，包含虚拟变量列)

**参数**：
- `column`: 源列名
- `mode`: 生成模式
  - `onehot` - 完整 one-hot 编码
  - `binary` - 单个二值列（指定值时为1）
- `value`: 触发值（binary 模式下，如 "United States"）
- `output_column`: 输出列名（如 "host_flag"）

**示例配置**：
```json
{
  "column": "NOC",
  "mode": "binary",
  "value": "United States",
  "output_column": "host_flag"
}
```

---

### 6. **data/map_values** - 值映射
**功能**：根据映射表替换列值

**输入**：
- Data (DataFrame)
- Mapping (dict 或 DataFrame)

**输出**：
- Result (DataFrame)

**参数**：
- `column`: 要映射的列
- `mapping_dict`: 直接指定映射（JSON格式，如 `{"2024": 329, "2028": 350}`）
- `default_value`: 未匹配时的默认值
- `output_column`: 输出列名

**示例配置**：
```json
{
  "column": "Year",
  "mapping_dict": "{\"2024\": 329}",
  "default_value": 300,
  "output_column": "events_total"
}
```

---

## 实现优先级

### Phase 1（立即实现）- 核心特征工程节点
1. ✅ `data/rolling_window` - 解决移动平均问题
2. ✅ `data/transform_column` - 解决差分/变换问题
3. ✅ `data/merge_dataframes` - 解决数据合并问题
4. ✅ `data/time_features` - 解决时间编码问题

### Phase 2（后续优化）- 扩展节点
5. `data/create_dummy` - 虚拟变量
6. `data/map_values` - 值映射
7. `data/pivot_table` - 数据透视
8. `data/conditional_column` - 条件生成列（if-else逻辑）
9. **`data/expression` - 表达式计算 (新)**

---

### 9. **data/expression** - 表达式计算
**功能**：使用类似 Matlab/Excel 的表达式创建新列或转换列

**输入**：
- Data (DataFrame)

**输出**：
- Result (DataFrame，包含新列)

**参数**：
- `expression`: 字符串格式的数学或逻辑表达式。列名可以直接使用。
- `output_column`: 输出列名

**支持的操作**：
- **算术**: `+`, `-`, `*`, `/`, `**`, `%`
- **比较**: `==`, `!=`, `>`, `<`, `>=`, `<=`
- **逻辑**: `&` (and), `|` (or), `~` (not)
- **数学函数**: `sin`, `cos`, `tan`, `log`, `log10`, `exp`, `sqrt`, `abs`
- **变量**: `@variable_name` (可引用外部 Python 变量)

**示例配置 1: 计算总奖牌**
```json
{
  "expression": "Gold + Silver + Bronze",
  "output_column": "total_medals"
}
```

**示例配置 2: 计算金牌占比**
```json
{
  "expression": "Gold / (Gold + Silver + Bronze)",
  "output_column": "gold_ratio"
}
```

**替代代码**：
```python
# 原自定义代码：
df["total_medals"] = df["Gold"] + df["Silver"] + df["Bronze"]
df["gold_ratio"] = df["Gold"] / df["total_medals"]

# 现在只需两个 data/expression 节点
# 技术实现: 后端使用 pandas.eval()，安全且高效
```

## 重构后的 MCM Problem C 工作流

### 原方案（1个巨大自定义节点）
```
[Load CSV] → [Load CSV] → [Load CSV] → [Custom Python 80行代码] → [Split] → [Regression]
```

### 新方案（无自定义节点）
```
[Load CSV: medals]
    ↓
[Filter Rows: Year >= 1988]
    ↓
[Rolling Window: gold_avg_3, groupby=NOC]
    ↓
[Rolling Window: gold_avg_5, groupby=NOC]
    ↓
[Transform Column: diff → gold_momentum]
    ↓
[Transform Column: fillna(0)]
    ↓
[Load CSV: hosts] → [Merge DataFrames: on=Year]
    ↓
[Create Dummy: host_flag]
    ↓
[Time Features: year_centered]
    ↓
[Select Column: X features] → [Split] → [Regression]
```

**优势**：
- ✅ 完全可视化，无需写代码
- ✅ 每个节点功能单一，易于调试
- ✅ 可复用到其他问题（如股票预测、天气预报）
- ✅ 易于理解工作流逻辑

---

## 技术实现要点

### 1. 节点属性面板增强
```javascript
// 支持下拉选择
{ type: "select", options: ["mean", "sum", "std"], default: "mean" }

// 支持多行文本（JSON输入）
{ type: "textarea", placeholder: '{"key": "value"}' }

// 支持列名自动补全（读取上游DataFrame的columns）
{ type: "column_selector", source: "input_0" }
```

### 2. 智能类型推断
- 节点自动检测上游是 DataFrame 还是 array
- 根据操作类型自动推断输出类型
- 在运行时进行类型检查与转换

### 3. 错误提示优化
```python
# 在生成器中添加友好的错误处理
try:
    medals["gold_avg_3"] = medals.groupby("NOC")["Gold"].rolling(3).mean()
except KeyError as e:
    raise ValueError(f"Column '{e}' not found in DataFrame. Available columns: {list(medals.columns)}")
```

---

## 预期效果

### 代码减少量
- **MCM Problem C**: 从 80 行自定义代码 → 0 行
- **一般时序预测**: 从 50 行 → 0 行
- **特征工程场景**: 平均减少 60-80% 自定义代码

### 用户体验提升
- **降低门槛**: 不懂 Python 的用户也能做复杂建模
- **提高效率**: 拖拽配置比写代码快 5-10 倍
- **减少错误**: 避免语法错误、类型错误、缩进问题

### 通用性保证
- 这些节点不是为 MCM 特制，而是通用数据科学操作
- 可应用于金融、医疗、制造业等任何领域
- 与 pandas/numpy 的标准 API 保持一致

---

## 下一步行动

1. ✅ **完成 Phase 1 四个核心节点的实现**（已完成设计）
2. 🔄 更新 `static/js/app.js` 注册节点
3. 🔄 更新 `app.py` 添加生成器函数
4. 🔄 编写单元测试验证每个节点
5. 📝 更新用户文档与示例
6. 🎯 用新节点重构 `mcm2025c_graph.json`，完全消除自定义节点

---

*设计文档版本: v1.0*  
*最后更新: 2025-12-03*
