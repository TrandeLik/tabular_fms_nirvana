"""YT row processor driven by a CatBoost ``cd`` spec.

Implements the :class:`yt_dataloader.table_processors.TableProcessor` contract
(``decode_fn`` + ``collate_fn``) that :class:`yt_dataloader.YTDataLoader` wraps
into a ``ytreader.CustomProcessor``. Each row is expected to carry a single
list-valued column (``features_column``) holding the full feature vector; the
:class:`~cd_utils.CdSpec` selects model features, the label, and the row key by
position within that list.
"""

from __future__ import annotations

import numpy as np

from cd_utils import CdSpec
from yt_dataloader.table_processors import TableProcessor


def _get(row: dict, key: str):
    """Fetch ``key`` from a YT row tolerating bytes-vs-str keys."""
    if key in row:
        return row[key]
    kb = key.encode()
    if kb in row:
        return row[kb]
    raise KeyError(
        f'Column {key!r} not found in row (available: '
        f'{[k.decode() if isinstance(k, bytes) else k for k in row]})'
    )


def _to_float_list(value) -> list[float]:
    """Coerce a YSON list column into a python list of floats."""
    if value is None:
        raise ValueError('Feature column is null')
    return [float(v) if v is not None else np.nan for v in value]


class DecodedRow(dict):
    """Lightweight decoded row: {'features': list, 'label': ..., 'docid': ...}."""


class CdTableProcessor(TableProcessor):
    def __init__(self, features_column: str, cd_spec: CdSpec) -> None:
        super().__init__()
        self.features_column = features_column
        self.cd_spec = cd_spec

    def decode_fn(self, row: dict) -> DecodedRow:
        raw = _to_float_list(_get(row, self.features_column))
        spec = self.cd_spec

        features = [raw[i] for i in spec.feature_indices]
        label = raw[spec.label_idx] if spec.label_idx is not None else None
        docid = raw[spec.docid_idx] if spec.docid_idx is not None else None

        return DecodedRow(features=features, label=label, docid=docid)

    def collate_fn(self, batch: list[DecodedRow]) -> dict:
        features = np.asarray([row['features'] for row in batch], dtype=np.float32)

        has_label = batch[0]['label'] is not None
        labels = (
            np.asarray([row['label'] for row in batch], dtype=np.float32)
            if has_label
            else None
        )

        has_docid = batch[0]['docid'] is not None
        docids = (
            np.asarray([row['docid'] for row in batch])  #
            if has_docid
            else None
        )

        return {'features': features, 'labels': labels, 'docids': docids}
