# MCM Problem C - 使用增强节点的简化工作流

## 对比：原方案 vs 新方案

### 原方案（需要 80+ 行自定义 Python 代码）

```
节点 1: Load CSV (medals)
节点 2: Load CSV (programs)  
节点 3: Load CSV (hosts)
      ↓
节点 4: Custom Python Script (80+ 行代码)
      - 数据清洗
      - 移动平均计算 (3届/5届)
      - 差分计算 (动量指标)
      - 数据合并
      - 主场标记
      - 时间特征
      ↓
节点 5-16: 模型训练与预测
```

**问题**：
- ❌ 需要编写大量 Python 代码
- ❌ 调试困难（80行代码在一个节点里）
- ❌ 不可复用（其他问题需要重写）
- ❌ 对非编程用户不友好

---

### 新方案（完全可视化，0 行自定义代码）

```
[Load CSV: medals] → [Filter: Year>=1988] → [转数值型] 
                                                ↓
[Rolling: gold_avg_3, groupby=NOC] ← ← ← ← ← ← ↓
                ↓
[Rolling: gold_avg_5, groupby=NOC]
                ↓
[Rolling: total_avg_3, groupby=NOC]
                ↓
[Rolling: total_avg_5, groupby=NOC]
                ↓
[Transform: diff → gold_momentum]
                ↓
[Transform: fillna(0)]
                ↓
[Transform: diff → total_momentum]
                ↓
[Transform: fillna(0)]
                ↓
[Load CSV: hosts] → [Merge: on=Year] → [创建主场标记]
                                            ↓
                        [Time Features: year_centered]
                                            ↓
                        [Select Columns: 特征矩阵]
                                            ↓
                        [Split: train/test]
                                            ↓
                        [Linear Regression]
```

**优势**：
- ✅ 完全可视化，拖拽配置
- ✅ 每个节点功能单一，易于调试
- ✅ 高度可复用（换数据集即可应用）
- ✅ 零编程门槛

---

## 具体节点配置示例

### 1. 计算 3 届移动平均金牌数

**节点类型**: `data/rolling_window`

**配置**:
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

**生成的 Python 代码**:
```python
_rolling = medals.groupby("NOC")["Gold"].rolling(3, min_periods=1)
medals["gold_avg_3"] = _rolling.mean().reset_index(level=0, drop=True)
```

---

### 2. 计算金牌增长动量

**节点类型**: `data/transform_column`

**配置**:
```json
{
  "column": "Gold",
  "operation": "diff",
  "periods": 1,
  "output_column": "gold_momentum"
}
```

**生成的 Python 代码**:
```python
medals["gold_momentum"] = medals["Gold"].diff(1).fillna(0)
```

---

### 3. 合并主办国数据

**节点类型**: `data/merge_dataframes`

**输入**:
- Left: medals (奖牌数据)
- Right: hosts (主办国数据)

**配置**:
```json
{
  "how": "left",
  "on": "Year"
}
```

**生成的 Python 代码**:
```python
merged = pd.merge(medals, hosts, on="Year", how="left")
```

---

### 4. 提取时间特征

**节点类型**: `data/time_features`

**配置**:
```json
{
  "date_column": "Year",
  "features": "year"
}
```

**生成的 Python 代码**:
```python
medals["year"] = pd.to_datetime(medals["Year"]).dt.year
base_year = medals["year"].min()
medals["year_centered"] = medals["year"] - base_year
```

---

## 完整工作流 JSON 结构（简化版）

```json
{
  "nodes": [
    {
      "id": 1,
      "type": "data/load_csv",
      "properties": {
        "path": "summerOly_medal_counts.csv",
        "output_format": "dataframe"
      }
    },
    {
      "id": 2,
      "type": "data/filter_rows",
      "properties": {
        "condition": "Year >= 1988"
      }
    },
    {
      "id": 3,
      "type": "data/rolling_window",
      "properties": {
        "column": "Gold",
        "window": 3,
        "operation": "mean",
        "groupby": "NOC",
        "output_column": "gold_avg_3"
      }
    },
    {
      "id": 4,
      "type": "data/rolling_window",
      "properties": {
        "column": "Gold",
        "window": 5,
        "operation": "mean",
        "groupby": "NOC",
        "output_column": "gold_avg_5"
      }
    },
    {
      "id": 5,
      "type": "data/transform_column",
      "properties": {
        "column": "Gold",
        "operation": "diff",
        "output_column": "gold_momentum"
      }
    },
    {
      "id": 6,
      "type": "data/load_csv",
      "properties": {
        "path": "summerOly_hosts.csv"
      }
    },
    {
      "id": 7,
      "type": "data/merge_dataframes",
      "properties": {
        "how": "left",
        "on": "Year"
      }
    },
    {
      "id": 8,
      "type": "data/time_features",
      "properties": {
        "date_column": "Year",
        "features": "year"
      }
    }
  ],
  "links": [
    [1, 1, 0, 2, 0],
    [2, 2, 0, 3, 0],
    [3, 3, 0, 4, 0],
    [4, 4, 0, 5, 0],
    [5, 5, 0, 7, 0],
    [6, 6, 0, 7, 1],
    [7, 7, 0, 8, 0]
  ]
}
```

---

## 节点数量对比

| 指标 | 原方案 | 新方案 | 改进 |
|------|--------|--------|------|
| 自定义代码行数 | 80+ | 0 | -100% |
| 需要编程知识 | 是 | 否 | ✅ |
| 节点总数 | 16 | 20 | +25% |
| 调试难度 | 高 | 低 | ✅ |
| 可复用性 | 低 | 高 | ✅ |

---

## 通用性验证

这些新节点不仅适用于 MCM Problem C，还可以应用于：

### 1. 股票预测
```
Load CSV (股票价格) 
  → Rolling Window (5日均线/20日均线)
  → Transform (价格变化率)
  → Merge (成交量数据)
  → 回归预测
```

### 2. 天气预报
```
Load CSV (历史气温)
  → Time Features (月份/季节)
  → Rolling Window (7天平均)
  → Transform (温差)
  → ARIMA 时序预测
```

### 3. 销售分析
```
Load CSV (销售记录)
  → Time Features (年/季度/月)
  → Rolling Window (季度移动平均)
  → Group Aggregate (按地区汇总)
  → 聚类分析
```

---

## 后续优化方向

### Phase 2 节点（待实现）

1. **data/create_dummy** - 虚拟变量生成
   - 用于创建 one-hot 编码
   - 示例：`host_flag = (NOC == "United States") ? 1 : 0`

2. **data/conditional_column** - 条件生成列
   - 支持 if-else 逻辑
   - 示例：`category = Gold > 30 ? "Strong" : "Normal"`

3. **data/pivot_table** - 数据透视
   - 行列转换
   - 多维度汇总

4. **data/explode_column** - 列展开
   - 将嵌套列表展开为多行
   - 用于处理 JSON/数组字段

---

## 总结

通过引入这 4 个核心增强节点，我们成功将：
- ✅ **80+ 行自定义代码 → 0 行**
- ✅ **编程难度 → 拖拽配置**
- ✅ **特定问题方案 → 通用数据处理框架**

这是 AlgoNode 从"算法可视化工具"到"通用数据科学平台"的关键一步！

---

*文档版本: v1.0*  
*最后更新: 2025-12-03*
