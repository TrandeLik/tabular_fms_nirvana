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

YT_TOKEN = os.environ.get('YT_TOKEN')


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cpu':
        return torch.device('cpu')
    return torch.device(
        f'cuda:{device_arg}' if torch.cuda.is_available() else 'cpu'
    )


def _load_mr_table(name: str) -> dict[str, str]:
    with open(name) as f:
        return json.load(f)


def _download_context(
    loader: YTDataLoader,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read the whole context loader into feature/label arrays."""
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    have_labels = True
    for batch in tqdm(loader, desc='context', unit='batch'):
        feats.append(batch['features'])
        if batch['labels'] is None:
            have_labels = False
        else:
            labels.append(batch['labels'])
    loader.reader.reader.close()

    x = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0) if have_labels and labels else None
    return x, y


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


def _to_output_rows(
    docids: np.ndarray | None,
    preds: np.ndarray,
    task_type: TaskType,
    offset: int,
) -> list[dict[str, Any]]:
    """Build YT output rows: a key plus the prediction.

    Classification writes the positive-class / argmax probability plus the full
    probability vector; regression writes the scalar value. When the table has
    no DocId, a running integer index is used as the key.
    """
    n = preds.shape[0]
    keys = (
        docids.tolist()
        if docids is not None
        else list(range(offset, offset + n))
    )
    rows: list[dict[str, Any]] = []
    if task_type == TaskType.REGRESSION:
        for k, v in zip(keys, preds.tolist()):
            rows.append({'key': _key(k), 'prediction': float(v)})
    elif task_type == TaskType.BINCLASS:
        pos = preds[:, 1] if preds.ndim == 2 else preds
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


def _key(k: Any) -> int:
    # DocId positions come through as floats (feature list is numeric); keys are
    # integer ids, so round-trip through int.
    return int(k)


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
        .add_column('key', ti.Int64)
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
    device = _resolve_device(args.device)
    logger.info(f'Device: {device}')

    copy_snapshot_to_out(str(CHECKPOINT_DIR))

    config = get_config(CONFIG_FILE)
    cd_spec = parse_cd(CD_FILE)
    task_type = TaskType(config.task_type)
    experiment_name = config.get_experiment_name()
    logger.info(f'Experiment: {experiment_name} | task_type={task_type.value}')

    context_mr = _load_mr_table('CONTEXT_MR_TABLE.json')
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
    if y_context_np is None:
        raise ValueError(
            'Context table has no Label column, but a label is required to '
            'build the ICL context (declare a Label in cd.txt).'
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
    have_test_labels = True
    offset = 0
    for batch in tqdm(test_loader, desc='predict', unit='batch'):
        x_batch = torch.from_numpy(transform_batch(batch['features'], fitted)).to(
            device
        )
        preds = predict_batch(model, ensemble, x_batch, _chunk_state=chunk_state)

        all_rows.extend(
            _to_output_rows(batch['docids'], preds, task_type, offset)
        )
        offset += preds.shape[0]

        if batch['labels'] is None:
            have_test_labels = False
        elif have_test_labels:
            y_true_parts.append(batch['labels'])
            y_pred_parts.append(preds)
    test_loader.reader.reader.close()

    # >>> Metrics (only if test labels are present)
    if have_test_labels and y_true_parts:
        y_true = np.concatenate(y_true_parts, axis=0)
        y_pred = np.concatenate(y_pred_parts, axis=0)
        if task_type == TaskType.BINCLASS and y_pred.ndim == 2:
            y_pred = y_pred[:, 1]
        metrics = calculate_metrics(
            y_true, y_pred, task_type, _prediction_type(task_type)
        )
        logger.info('Test metrics:')
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        logger.info('Test table has no labels — writing predictions only.')

    # >>> Write predictions to YT
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
    with open('MR_TABLE_OUTPUT', 'w') as f:
        json.dump(mr_out, f)
    logger.info(f'Done. Output: {mr_out}')


if __name__ == '__main__':
    main()
