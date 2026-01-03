# Glucometric prediction pipeline

## Requirements

- Python 3.11 or 3.12
- Poetry

## Installation

```bash
git clone https://github.com/jamestq/sucre
cd sucre
poetry install
poetry shell
```

## Usage

```bash
sucre run <config-file.yaml>
```

## Configuration

### Preprocessing

```yaml
data_folder: .

combine_data:
  output: combined.xlsx
  inputs:
  - path: data_source.xlsx
    tab: 0
    drop: [column_to_drop]
    keep: [column_to_keep]

filter_data:
  input: combined.xlsx
  output: filtered.xlsx
  filters:
  - column: group_column
    operation: "=="  # ==, !=, >, <, >=, <=
    value: "Control"

encode_data:
  input: filtered.xlsx
  output: encoded.xlsx
  encodings:
  - columns: [binary_outcome]
    default: 0
    conditions:
    - operation: ">="
      value: 1
      encoded_value: 1

impute_data:
  input: encoded.xlsx
  output: imputed.xlsx
  default_method: median  # median, mean, mode, iterative
  imputations:
  - columns: [column_with_missing]
    method: iterative
```

### Training

```yaml
data_folder: .

train:
  input: prepared_data.xlsx
  output: training_results
  drop: [index]
  targets: [outcome_1, outcome_2]
  normalize: [zscore, minmax, maxabs, robust]
  transform: [quantile]
  models: [rf, neural_network, lightgbm, rbfsvm]
```

## Available Options

| Normalization | Description |
|---------------|-------------|
| `zscore` | Zero mean, unit variance |
| `minmax` | Scale to 0-1 |
| `maxabs` | Scale to -1 to 1 |
| `robust` | Quartile-based (outlier robust) |

| Model | Description |
|-------|-------------|
| `rf` | Random Forest |
| `lightgbm` | LightGBM |
| `rbfsvm` | SVM (RBF kernel) |
| `neural_network` | Neural Network |

Additional PyCaret models are also supported.

## Custom Preprocessing

```yaml
custom_operation:
  path: custom_script.py
  input: data.xlsx
  output: processed.xlsx
```

The script must define a function matching the operation name.

## Output Structure

```
training_results/
├── 0/
│   └── target_name/
│       └── model_name/
│           └── transformer/
│               ├── training.xlsx
│               ├── prediction.xlsx
│               ├── metrics.xlsx
│               ├── Confusion Matrix.png
│               ├── AUC.png
│               └── PR Curve.png
└── training_log.txt
```
