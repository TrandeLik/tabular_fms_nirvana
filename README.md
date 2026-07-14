# tabular_fms_nirvana

Run a **Tabular Foundation Model (TFM)** on YT tables, in the spirit of applying
CatBoost — but instead of training, the model does *in-context learning*: a
**context** table is used as the ICL support set, and predictions are produced
for a **test** table. Designed to run as a Nirvana operation.

Currently supported model: **TabICL** (`tfm_name: tabicl`).

## How it works

1. **Download** the whole `CONTEXT` table → ICL context `(x_train, y_train)`.
2. **Fit** feature normalization (numerical + categorical) on the context.
3. **Stream** the `TEST` table in batches; predict each batch with the TFM
   (feature-permutation ensembling + chunked forward with CUDA-OOM halving).
4. **Write** predictions back to a YT table.
5. If the test rows carry a label, **compute and log metrics** to stdout.

The context table must have a `Label`; the test table's label is optional
(metrics are logged only when present).

## Inputs (operation working directory)

| File | Purpose |
|------|---------|
| `CONFIG.yaml` | Run configuration — see `config.py::Config`. |
| `cd.txt` | CatBoost-style column description — see below. |
| `TRAIN_MR_TABLE.json` | `{"cluster": ..., "table": ...}` — ICL context table (downloaded fully). |
| `TEST_MR_TABLE.json` | `{"cluster": ..., "table": ...}` — table to predict (streamed). |

`YT_TOKEN` must be set in the environment.

### `cd.txt` — column description

A [CatBoost-style](https://catboost.ai/docs/en/concepts/input-data_column-descfile)
column description. Columns are numbered from 0 as in the original dataset;
values reach the job from a YT table in two places:

- **`SampleId` / `DocId`** — a **separate YT column** (addressed by the name in
  the cd line's 3rd field), holding the row key written next to each prediction.
  It is **not** part of the feature list. **Required.**
- Every other column lives inside a single **list-valued** YT column (name
  configurable via `features_column`, default `features`), with **`Label`** as
  its **first** element, followed by the feature values.

```
0	SampleId	key                          # separate YT column 'key'
1	Label	is_fraud                         # features[0]
2	Num	FActiveDayPercent(Record)            # features[1]
3	Num	FBaseAntifraudAction1d(DeviceId)     # features[2]
```

Tab-separated `<index>\t<Type>[\t<NAME>]`. `#` comments and blank lines are
ignored. A column at cd index `i` maps to feature-list position `i − <Label
index>` (so `Label` → `features[0]`). Recognized roles:

- **`Label`** — the target (`y`), the first element of the feature list. Used
  for the ICL context; on test, metrics are computed over rows whose label is
  present (non-NaN). **Required.**
- **`SampleId` / `DocId`** — the separate id column described above. **Required.**
- **`Num`** — numerical feature (normalized).
- **`Categ` / `Categorical`** — categorical feature (encoded).
- **`Weight`** — parsed but currently ignored.
- **Anything else** (`Auxiliary`, `Text`, `GroupId`, …) after `Label` —
  **excluded** from the feature matrix. The parser never errors on an unknown
  feature type.

Only `Num` + `Categ` columns form the matrix fed to the model, in `cd` order.
`Label` and `SampleId`/`DocId` are required; the parser raises otherwise.

### `CONFIG.yaml`

```yaml
# Model
tfm_name: tabicl            # currently only 'tabicl' is supported
tfm_config: {}              # extra kwargs forwarded to the wrapper
task_type: binclass         # regression | binclass | multiclass

# Data
features_column: features   # name of the list-valued YT column
batch_size: 1024
output_table_tmp_path: "//home/yr/trandelik/crypta/datasets/"

# Feature normalization (fit on context, reused per batch)
num_policy: standard        # null | standard | quantile-normal | quantile-uniform
cat_policy: ordinal         # null | ordinal | standard | one-hot
impute_strategy: basic      # null | basic | standardize_min

# Inference
seed: 0
max_context_size: null      # subsample context to at most N rows (null = all)
n_ensemble: 1               # feature-permutation + context-subsample ensembling
eval_chunk_size: null       # rows per forward pass (null = whole batch)
```

## Output

A temporary YT table under `output_table_tmp_path`, with schema:

| column | type | meaning |
|--------|------|---------|
| `key` | `Int64` | the `DocId` (or running index) |
| `prediction` | `Double` | regression value / positive-class prob (binclass) / argmax class (multiclass) |
| `probabilities` | `Optional[List<Double>]` | class probabilities (classification only) |

The descriptor `{"cluster", "table"}` of the output table is written to
`MR_TABLE_OUTPUT` in the working directory. When the test table has a label,
metrics (from `lib.metrics`) are also printed to stdout.

## Running

```bash
# Populate CONFIG.yaml, cd.txt, CONTEXT_MR_TABLE.json, TEST_MR_TABLE.json first.
export YT_TOKEN=...
uv run --group yandex python main_yt.py --proxy hahn
```

CLI flags: `--proxy <cluster>` (output cluster, default `hahn`),
`--device cpu|<cuda-index>` (default: auto-detect), `--debug` (skip the YT
upload).

## Layout

```
main_yt.py           entry point (setup → context → stream+predict → write → metrics)
config.py            Config dataclass + get_config / parse_args
cd_utils.py          CatBoost cd.txt parser
table_processor.py   YT row → (features, label, docid) decoder/collator
feature_pipeline.py  sklearn normalization fit on context, reused per batch
inference.py         ICL core: ContextEnsemble + predict_batch (chunking, OOM, ensembling)
lib/                 TFM wrappers (lib.tfm.tabicl), metrics, utils
yt_dataloader/       streaming YT table reader
nirvana_stuff/       Nirvana snapshot + YT output helpers
```
