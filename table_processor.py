"""YT row processor driven by a CatBoost ``cd`` spec.

Implements the :class:`yt_dataloader.table_processors.TableProcessor` contract
(``decode_fn`` + ``collate_fn``) that :class:`yt_dataloader.YTDataLoader` wraps
into a ``ytreader.CustomProcessor``. Each row carries:

* a single **list-valued** column (``features_column``) whose first element is
  the ``Label`` and whose remaining elements are the feature values, and
* a separate **id** column (``cd_spec.docid_column``) holding the row key.

The :class:`~cd_utils.CdSpec` selects model features and the label by position
within the list, and names the id column.
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


def _to_float_list(value, column: str) -> list[float]:
    """Coerce a YSON list column into a python list of floats."""
    if value is None:
        raise ValueError(f'Feature column {column!r} is null')
    if isinstance(value, (str, bytes)) or not hasattr(value, '__iter__'):
        raise TypeError(
            f'Feature column {column!r} must be a list, got {type(value).__name__}'
        )
    out: list[float] = []
    for i, v in enumerate(value):
        if v is None:
            out.append(np.nan)
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError) as err:
            raise ValueError(
                f'Feature column {column!r} position {i} is not numeric: {v!r}'
            ) from err
    return out


class DecodedRow(dict):
    """Lightweight decoded row: {'features': list, 'label': float, 'docid': Any}."""


class CdTableProcessor(TableProcessor):
    def __init__(self, features_column: str, cd_spec: CdSpec) -> None:
        super().__init__()
        if not features_column:
            raise ValueError('features_column must be a non-empty string')
        self.features_column = features_column
        self.cd_spec = cd_spec

    def decode_fn(self, row: dict) -> DecodedRow:
        spec = self.cd_spec
        raw = _to_float_list(_get(row, self.features_column), self.features_column)

        # The cd file references positions inside the feature list; a row shorter
        # than expected means the cd file and the table are out of sync.
        if len(raw) <= spec.max_feature_pos:
            raise ValueError(
                f'Row has {len(raw)} values in column {self.features_column!r}, '
                f'but cd.txt references feature position {spec.max_feature_pos}. '
                f'The cd file and the table schema are out of sync.'
            )

        features = [raw[i] for i in spec.feature_positions]
        label = raw[spec.label_pos]
        # The id is a separate YT column, kept raw (it is a key, not a feature).
        docid = _get(row, spec.docid_column)

        return DecodedRow(features=features, label=label, docid=docid)

    def collate_fn(self, batch: list[DecodedRow]) -> dict:
        if not batch:
            raise ValueError('collate_fn received an empty batch')
        features = np.asarray([row['features'] for row in batch], dtype=np.float32)
        # Label always exists (required by the cd); unlabeled rows carry NaN.
        labels = np.asarray([row['label'] for row in batch], dtype=np.float32)
        docids = np.asarray([row['docid'] for row in batch])
        return {'features': features, 'labels': labels, 'docids': docids}
