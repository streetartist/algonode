# Enhanced Data Nodes - Design Notes

## Purpose
- Reduce custom scripting by covering the most common feature-engineering patterns for mathematical modeling.
- Keep front-end node UX simple (few, well-named properties) while generating robust pandas/NumPy code on the backend.
- Stay composable with the existing LiteGraph pipeline and code generator without breaking older flows.

## Scope
The enhanced nodes target tabular and time-based modeling needs:
- Rolling statistics, column transforms, joins, and time features.
- Encoding, mapping, conditional logic, pivoting, row filtering, and lightweight expressions.
- Output remains a DataFrame by default but can feed matrix/array consumers through downstream selectors.

## Architecture (frontend ↔ backend)
- Frontend registration: `static/js/app.js` calls `registerNode(...)` with property defaults and widgets for the UX.
- Backend generation: `app.py` maps node types to code emitters in `generate_scope`:
  - `data/rolling_window` → `gen_rolling_window`
  - `data/transform_column` → `gen_transform_column`
  - `data/merge_dataframes` → `gen_merge_dataframes`
  - `data/time_features` → `gen_time_features`
  - `data/create_dummy` → `gen_create_dummy`
  - `data/map_values` → `gen_map_values`
  - `data/conditional_column` → `gen_conditional_column`
  - `data/pivot_table` → `gen_pivot_table`
  - `data/explode_column` → `gen_explode_column`
  - `data/expression` → `gen_expression`
- Safety: all generators coerce inputs to DataFrame (`pd.DataFrame(...)`) when needed and guard with try/except to avoid hard crashes.
- Naming: `sanitize_name` ensures python-safe variable names; outputs default to `{column}_{op}` when user leaves blanks.

## Node contracts
| Node | Inputs | Key properties | Output | Notes |
| --- | --- | --- | --- | --- |
| Rolling Window | DataFrame | column, window, operation(mean/sum/std/min/max/median), groupby (csv), min_periods, output_column | DataFrame | Grouped rolling uses `reset_index(level=0, drop=True)` to align index. |
| Transform Column | DataFrame | column, operation(diff/pct_change/shift/cumsum/log/sqrt/abs/fillna/round), periods/decimals/fill_value, groupby, output_column | DataFrame | Group-aware ops use `groupby().<op>`; fillna stays per-column for clarity. |
| Merge DataFrames | left, right | how(inner/left/right/outer), on or left_on/right_on | DataFrame | Falls back to default merge when keys omitted. |
| Time Features | DataFrame | date_column, features list (year, month, day, dayofweek, quarter, dayofyear, weekofyear) | DataFrame | Uses `pd.to_datetime(..., errors='coerce')` before extraction. |
| Create Dummy | DataFrame | column, mode(onehot/binary), value/output_column (binary), prefix (onehot) | DataFrame | One-hot drops source column after concat to avoid duplication. |
| Map Values | DataFrame | column, mapping_dict stringified JSON, default_value, output_column | DataFrame | Leaves original column untouched; writes mapped values separately. |
| Conditional Column | DataFrame | condition expression, true_value, false_value, output_column | DataFrame | Uses `eval`; masks default to `False` on evaluation errors. |
| Pivot Table | DataFrame | index, columns, values, aggfunc, fill_value, margins | DataFrame | Supports blank index/columns for quick totals. |
| Explode Column | DataFrame | column, output_column, ignore_index | DataFrame | Resets index when `ignore_index=true`. |
| Expression | DataFrame | expression using column names (A,B,...), output_column | DataFrame | Evaluated via `DataFrame.eval`; keeps inputs untouched. |

## Data model & interoperability
- Preferred hand-off type is pandas DataFrame; nodes that require arrays (e.g., modeling or metrics) should be preceded by `data/select_column` or converters already present in the library.
- Output format toggles (`output_format`) are preserved for legacy nodes; new nodes keep DataFrame to retain column names.
- Graph traversal order is topological; multi-output nodes return dicts keyed by slot index where needed.

## Error handling & fallbacks
- All emitters wrap risky operations with guarded defaults (empty arrays/DataFrames) to keep code generation resilient.
- Missing columns trigger warnings in generated code instead of raising; downstream nodes still execute with best-effort data.

## Extensibility guidelines
- Add new data nodes by pairing a `registerNode` call (frontend) with a `gen_*` emitter and mapping entry in `app.py`.
- Keep property names short and explicit; prefer strings/ints over nested config to simplify widget auto-binding.
- Provide deterministic defaults so exported scripts run without manual edits.

## Validation strategy
- Unit-level: compare generated scripts for each node type with small fixtures (grouped vs non-grouped, empty data).
- Integration: run the `examples/8_demo.json` (Olympics) graph to cover rolling, transforms, merge, and time features end-to-end.
- Backward compatibility: load older JSON graphs to ensure unknown properties are ignored gracefully.
