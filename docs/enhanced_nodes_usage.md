# Enhanced Nodes - Usage Guide

This guide shows how to apply the strengthened data nodes to common modeling tasks without writing custom code.

## Quickstart (5 steps)
1. Run `python app.py` and open `http://localhost:5000`.
2. Drag nodes from **Data预处理** and **数据与输入** to build your pipeline.
3. Configure properties in the node body (text/number widgets already bound to code generation).
4. Click **Run Model** to execute; use **Output** node to inspect results in the browser or export Python code.
5. Save the canvas as JSON to reuse or version your workflow.

## Node recipes
### Rolling Window (`data/rolling_window`)
- Typical use: smoothing metrics or computing grouped moving averages.
- Inputs: DataFrame with target column.
- Set `column`, `window`, `operation` (`mean`, `sum`, `std`, `min`, `max`, `median`), optional `groupby` (`Team,Year`) and `min_periods`.
- Leave `output_column` blank to auto-name as `{column}_{operation}_{window}`.

### Transform Column (`data/transform_column`)
- Use for differencing, percent change, shifts, cumulative sums, log/sqrt/abs, fillna, or rounding.
- Key props: `column`, `operation`, `periods` (for diff/pct_change/shift), `fill_value`, `decimals`, optional `groupby`.
- Output writes to `output_column` (auto `{column}_{operation}` if blank) without overwriting source unless you choose the same name.

### Merge DataFrames (`data/merge_dataframes`)
- Inputs: `Left`, `Right`. Set `how` (`inner`, `left`, `right`, `outer`).
- Choose either shared `on` key or `left_on`/`right_on` pair. Defaults to pandas merge when keys are empty.

### Time Features (`data/time_features`)
- Converts `date_column` to datetime and expands selected features (`year,month,day,dayofweek,quarter,dayofyear,weekofyear`).
- Useful for ARIMA/ML models that need calendar signals.

### Create Dummy (`data/create_dummy`)
- Mode `onehot`: expands categorical column with `prefix` (defaults to column name) and drops the original.
- Mode `binary`: set `value` and `output_column` for a single indicator.

### Conditional Column (`data/conditional_column`)
- Enter a pandas-style expression in `condition` (e.g., `Gold > 0 & Silver > 0`).
- Fills `true_value`/`false_value` into `output_column`.

### Pivot Table (`data/pivot_table`)
- Set `index`, `columns`, `values`, `aggfunc` (e.g., `mean`, `sum`), optional `fill_value`, `margins`.
- Keeps DataFrame shape; downstream `select_column` can turn slices into arrays.

### Explode Column (`data/explode_column`)
- Use when a cell contains lists/strings to be expanded into rows. `ignore_index=true` to reindex after explosion.

### Expression (`data/expression`)
- Lightweight arithmetic without writing a Python node. Example: `0.6 * A + 0.4 * B` → `result` column.

## End-to-end workflow examples
- Olympics momentum (enhanced demo): `examples/8_demo.json` chains Load CSV → Filter Rows → Rolling Window (3,5) → Transform (diff/fillna) → Merge Hosts → Time Features → Output.
- Feature engineering starter: Load CSV → Create Dummy (categorical) → Transform Column (log/round) → Pivot Table → Select Column → Model/Metric.

## Tips for robust pipelines
- Keep date columns in ISO strings before **Time Features**; the generator uses `errors='coerce'` to avoid crashes.
- For grouped ops, provide comma-separated group keys and ensure they exist; missing keys degrade gracefully but may warn.
- If a downstream model expects arrays, place **Select Column** or **Describe** to inspect shapes before training.
- Reuse **Custom Python Script** only for edge cases; most pandas transforms are covered by the enhanced nodes to keep graphs readable.
