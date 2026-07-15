"""Entry point: run a Tabular Foundation Model on YT tables.

Usage pattern (Nirvana): the operation is given, in its working directory,
``CONFIG.yaml`` (see :mod:`config`), ``cd.txt`` (see :mod:`cd_utils`), and two
MR-table descriptors ``CONTEXT_MR_TABLE.json`` / ``TEST_MR_TABLE.json`` (each
``{"cluster": ..., "table": ...}``).

Flow (analogous to ``main.py`` but YT-native and non-graph):
1. Download the CONTEXT table fully -> ICL context (x_train, y_train).
2. Fit feature normalization on the context.
3. Stream the TEST table in batches; predict each batch with the TFM
   (ensembling + chunked forward + CUDA-OOM halving).
4. Write predictions back to a YT table.
5. If the test rows carry a Label, compute and log metrics to stdout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import yt.wrapper as yt
import yt.type_info.typing as ti

sys.path.append('./')

import lib  # noqa: E402  (configures warnings)
from cd_utils import parse_cd  # noqa: E402
from config import get_config, parse_args  # noqa: E402
from inference import ContextEnsemble, predict_batch  # noqa: E402
from feature_pipeline import fit_transform_context, transform_batch  # noqa: E402
from lib.metrics import calculate_metrics  # noqa: E402
from lib.tfm import load_tfm  # noqa: E402
from lib.util import TaskType  # noqa: E402
from nirvana_stuff import copy_snapshot_to_out  # noqa: E402
from table_processor import CdTableProcessor  # noqa: E402
from yt_dataloader import YTDataLoader  # noqa: E402

CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'
CD_FILE = Path('cd.txt')
CONFIG_FILE = Path('CONFIG.yaml')
# Descriptor of the written output table ({"cluster", "table"}), consumed by
# downstream Nirvana operations.
OUTPUT_MR_TABLE_FILE = Path('MR_TABLE_OUTPUT.json')

# Classification sentinel written into the Label slot for rows that have no
# target. Such rows are skipped when building the ICL context (they cannot serve
# as support) and are excluded from metrics; regression uses NaN for the same.
NO_LABEL_CLASS = -9999

YT_TOKEN = os.environ.get('YT_TOKEN')


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cpu':
        return torch.device('cpu')
    # An explicit CUDA index was requested; refuse to silently fall back to CPU.
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'--device {device_arg} requests a GPU, but CUDA is not available. '
            f'Pass --device cpu to run on CPU.'
        )
    try:
        index = int(device_arg)
    except ValueError as err:
        raise ValueError(
            f"--device must be 'cpu' or a CUDA index, got {device_arg!r}"
        ) from err
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f'--device {index} is out of range; {torch.cuda.device_count()} '
            f'CUDA device(s) visible'
        )
    return torch.device(f'cuda:{index}')


def _load_mr_table(name: str) -> dict[str, str]:
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(
            f'MR table descriptor {name} not found in the working directory'
        )
    try:
        mr = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise ValueError(f'{name} is not valid JSON: {err}') from err
    if not isinstance(mr, dict) or 'table' not in mr:
        raise ValueError(
            f'{name} must be a JSON object with at least a "table" key '
            f'(and usually "cluster"), got {mr!r}'
        )
    if not mr['table']:
        raise ValueError(f'{name} has an empty "table" value')
    return mr


def _download_context(
    loader: YTDataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the whole context loader into feature/label arrays.

    Label is always present (required by the cd file), so labels are returned
    unconditionally; unlabeled rows would carry NaN.
    """
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in tqdm(loader, desc='context', unit='batch'):
        feats.append(batch['features'])
        labels.append(batch['labels'])
    loader.reader.reader.close()

    if not feats:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _make_loader(
    table: str,
    cluster: str,
    client: yt.YtClient,
    config,
    cd_spec,
    batch_size: int,
) -> YTDataLoader:
    return YTDataLoader(
        table_name=table,
        batch_size=batch_size,
        num_table_readers=2,
        num_subprocesses=4,
        cache_size=2048,
        queue_size_limit=1024,
        client=client,
        cluster=cluster,
        processor=CdTableProcessor(config.features_column, cd_spec),
    )


def _prediction_type(task_type: TaskType) -> str:
    return 'labels' if task_type == TaskType.REGRESSION else 'probs'


def _labeled_mask(labels: np.ndarray, task_type: TaskType) -> np.ndarray:
    """Boolean mask of rows that carry a usable target.

    A row has no target when its label is NaN (missing everywhere) or, for
    classification, when it equals the ``-9999`` sentinel. Such rows cannot
    serve as ICL support and are excluded from metrics.
    """
    mask = ~np.isnan(labels)
    if task_type != TaskType.REGRESSION:
        mask &= labels != NO_LABEL_CLASS
    return mask


def _to_output_rows(
    docids: np.ndarray,
    preds: np.ndarray,
    task_type: TaskType,
) -> list[dict[str, Any]]:
    """Build YT output rows: the row key (SampleId) plus the prediction.

    Classification writes the positive-class / argmax probability plus the full
    probability vector; regression writes the scalar value.
    """
    if docids.shape[0] != preds.shape[0]:
        raise ValueError(
            f'docids ({docids.shape[0]}) and predictions ({preds.shape[0]}) '
            f'length mismatch'
        )
    keys = docids.tolist()
    rows: list[dict[str, Any]] = []
    if task_type == TaskType.REGRESSION:
        for k, v in zip(keys, preds.tolist()):
            rows.append({'key': _key(k), 'prediction': float(v)})
    elif task_type == TaskType.BINCLASS:
        if preds.ndim != 2:
            pos = preds
        elif preds.shape[1] >= 2:
            pos = preds[:, 1]
        else:
            # Context held a single class, so the model emitted an (N, 1)
            # column; the positive class was never observed. Column 0 is
            # P(the only class seen) — report its complement as P(positive).
            pos = 1.0 - preds[:, 0]
        for k, p in zip(keys, pos.tolist()):
            rows.append(
                {'key': _key(k), 'prediction': float(p), 'probabilities': [float(p)]}
            )
    else:  # MULTICLASS
        argmax = preds.argmax(axis=1)
        for k, cls, probs in zip(keys, argmax.tolist(), preds.tolist()):
            rows.append(
                {
                    'key': _key(k),
                    'prediction': float(cls),
                    'probabilities': [float(x) for x in probs],
                }
            )
    return rows


def _key(k: Any) -> int | str:
    """Coerce a SampleId value to an output key, preserving its natural type.

    Ids arrive from a dedicated YT column. They are commonly integers, but may
    also be arbitrary **string** keys (e.g. resource paths like
    ``'/users/.../feedback'``); both are valid and preserved as-is. Whole-valued
    floats are narrowed to int; a genuinely fractional float is a schema
    mismatch worth surfacing rather than silently truncating.
    """
    if isinstance(k, bool):
        raise TypeError(f'SampleId must be an int or string, got bool {k!r}')
    if isinstance(k, (int, np.integer)):
        return int(k)
    if isinstance(k, (float, np.floating)):
        if not float(k).is_integer():
            raise ValueError(f'SampleId {k!r} is a non-integer float')
        return int(k)
    if isinstance(k, (str, bytes)):
        return k.decode() if isinstance(k, bytes) else k
    raise TypeError(f'Unsupported SampleId type {type(k).__name__}: {k!r}')


def _key_column_type(rows: list[dict[str, Any]]):
    """Pick the YT type for the ``key`` column from the actual keys.

    All SampleIds in a table are expected to share a type. String keys -> String,
    integer keys -> Int64. A mix is a schema inconsistency worth surfacing.
    """
    has_str = any(isinstance(r['key'], str) for r in rows)
    has_int = any(isinstance(r['key'], int) for r in rows)
    if has_str and has_int:
        raise ValueError(
            'SampleId keys mix string and integer types within one table; '
            'the output key column requires a single type.'
        )
    return ti.String if has_str else ti.Int64


def _write_predictions(
    rows: list[dict[str, Any]],
    client: yt.YtClient,
    cluster: str,
    table_root: str,
    name_prefix: str,
) -> dict[str, str]:
    from random import choices
    from string import ascii_lowercase

    suffix = ''.join(choices(ascii_lowercase, k=10))
    out_table = os.path.join(table_root, f'{name_prefix}_{suffix}')

    schema = (
        yt.schema.TableSchema()
        .add_column('key', _key_column_type(rows))
        .add_column('prediction', ti.Double)
        .add_column('probabilities', ti.Optional[ti.List[ti.Double]])
    )
    logger.info(f'Writing {len(rows)} predictions to {out_table}')
    yt.create('table', out_table, attributes={'schema': schema}, client=client)
    yt.write_table(out_table, rows, client=client)
    return {'cluster': cluster, 'table': out_table}


def main() -> None:
    # >>> Setup
    args = parse_args()
    if not YT_TOKEN:
        raise RuntimeError(
            'YT_TOKEN environment variable is not set; it is required to access YT.'
        )
    device = _resolve_device(args.device)
    logger.info(f'Device: {device}')

    copy_snapshot_to_out(str(CHECKPOINT_DIR))

    config = get_config(CONFIG_FILE)
    cd_spec = parse_cd(CD_FILE)
    task_type = TaskType(config.task_type)
    experiment_name = config.get_experiment_name()
    logger.info(f'Experiment: {experiment_name} | task_type={task_type.value}')

    context_mr = _load_mr_table('TRAIN_MR_TABLE.json')
    test_mr = _load_mr_table('TEST_MR_TABLE.json')

    yt.config.config['token'] = YT_TOKEN

    # Each MR table declares its own cluster; bind a client per cluster so both
    # streaming reads and row-count/schema lookups hit the right proxy. Output
    # is written to the proxy passed on the command line.
    context_cluster = context_mr.get('cluster', args.proxy)
    test_cluster = test_mr.get('cluster', args.proxy)
    context_client = yt.YtClient(proxy=context_cluster, token=YT_TOKEN)
    test_client = yt.YtClient(proxy=test_cluster, token=YT_TOKEN)
    out_client = yt.YtClient(proxy=args.proxy, token=YT_TOKEN)

    # >>> Download context fully -> ICL context
    logger.info(f'Downloading context table {context_mr["table"]}')
    context_loader = _make_loader(
        context_mr['table'],
        context_cluster,
        context_client,
        config,
        cd_spec,
        config.batch_size,
    )
    raw_context, y_context_np = _download_context(context_loader)
    if raw_context.shape[0] == 0:
        raise ValueError(
            f'Context table {context_mr["table"]!r} is empty; cannot build an '
            f'ICL context.'
        )
    # Only labeled rows can serve as ICL support; drop rows with no target
    # (NaN, or the -9999 sentinel for classification).
    labeled = _labeled_mask(y_context_np, task_type)
    n_dropped = int((~labeled).sum())
    if n_dropped:
        logger.warning(f'Dropping {n_dropped} unlabeled context rows (no target)')
        raw_context = raw_context[labeled]
        y_context_np = y_context_np[labeled]
    if raw_context.shape[0] == 0:
        raise ValueError(
            f'Context table {context_mr["table"]!r} has no labeled rows; cannot '
            f'build an ICL context.'
        )
    logger.info(f'Context: {raw_context.shape[0]} rows, {raw_context.shape[1]} feats')

    # >>> Fit normalization on the context, move to device
    x_context_np, fitted = fit_transform_context(raw_context, cd_spec, config)
    x_context = torch.from_numpy(x_context_np).to(device)
    if task_type == TaskType.REGRESSION:
        y_context = torch.from_numpy(y_context_np.astype(np.float32)).to(device)
    else:
        y_context = torch.from_numpy(y_context_np.astype(np.int64)).to(device)

    # >>> Model
    model = load_tfm(name=config.tfm_name, device=device, config=config.tfm_config)
    model.to(device)
    model.eval()
    logger.info(f'Loaded TFM {config.tfm_name!r}')

    ensemble = ContextEnsemble(
        x_context,
        y_context,
        task_type=task_type,
        n_ensemble=config.n_ensemble,
        max_context_size=config.max_context_size,
        seed=config.seed,
    )

    # >>> Stream test, predict batch-by-batch
    logger.info(f'Streaming test table {test_mr["table"]}')
    test_loader = _make_loader(
        test_mr['table'],
        test_cluster,
        test_client,
        config,
        cd_spec,
        config.batch_size,
    )
    chunk_state = [config.eval_chunk_size or config.batch_size]

    all_rows: list[dict[str, Any]] = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    for batch in tqdm(test_loader, desc='predict', unit='batch'):
        x_batch = torch.from_numpy(transform_batch(batch['features'], fitted)).to(
            device
        )
        preds = predict_batch(model, ensemble, x_batch, _chunk_state=chunk_state)

        all_rows.extend(_to_output_rows(batch['docids'], preds, task_type))

        # Label is always present in the row (required by the cd) but may be NaN
        # for unlabeled test rows; collect everything and filter NaN below.
        y_true_parts.append(batch['labels'])
        y_pred_parts.append(preds)
    test_loader.reader.reader.close()

    # >>> Metrics over the rows that carry a target only (NaN, and the -9999
    # classification sentinel, are excluded).
    y_true = np.concatenate(y_true_parts, axis=0)
    y_pred = np.concatenate(y_pred_parts, axis=0)
    labeled = _labeled_mask(y_true, task_type)
    n_labeled = int(labeled.sum())
    if n_labeled:
        y_true = y_true[labeled]
        y_pred = y_pred[labeled]
        if task_type == TaskType.BINCLASS and y_pred.ndim == 2:
            y_pred = y_pred[:, 1]
        metrics = calculate_metrics(
            y_true, y_pred, task_type, _prediction_type(task_type)
        )
        logger.info(f'Test metrics ({n_labeled} labeled rows):')
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        logger.info('Test table has no labeled rows — writing predictions only.')

    # >>> Write predictions to YT
    if not all_rows:
        raise ValueError(
            f'Test table {test_mr["table"]!r} produced no rows; nothing to write.'
        )
    if args.debug:
        logger.info('Debug mode: skipping upload to YT.')
        return
    mr_out = _write_predictions(
        all_rows,
        out_client,
        args.proxy,
        config.output_table_tmp_path,
        experiment_name,
    )
    with OUTPUT_MR_TABLE_FILE.open('w') as f:
        json.dump(mr_out, f)
    logger.info(f'Done. Output: {mr_out}')


if __name__ == '__main__':
    main()
