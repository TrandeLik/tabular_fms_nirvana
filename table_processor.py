"""YT row processor driven by a CatBoost ``cd`` spec.

Implements the :class:`yt_dataloader.table_processors.TableProcessor` contract
(``decode_fn`` + ``collate_fn``) that :class:`yt_dataloader.YTDataLoader` wraps
into a ``ytreader.CustomProcessor``. Each row carries:

* a single **bytes** column (``features_column``) holding a tab-separated string
  of float values; decoded and split on ``\t`` it yields a vector whose first
  element is the ``Label`` and whose remaining elements are the feature values,
  and
* a separate **id** column (``cd_spec.docid_column``) holding the row key.

The :class:`~cd_utils.CdSpec` selects model features and the label by position
within the decoded vector, and names the id column.
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


def _to_float_array(value, column: str) -> np.ndarray:
    """Decode a tab-separated bytes/str feature column into a float32 array.

    The column arrives as ``bytes`` (or ``str``); its decoded content is a
    ``\\t``-separated list of float values whose positions are addressed by the
    cd spec. Empty fields become NaN.
    """
    if value is None:
        raise ValueError(f'Feature column {column!r} is null')
    if isinstance(value, bytes):
        text = value.decode()
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError(
            f'Feature column {column!r} must be bytes/str, got '
            f'{type(value).__name__}'
        )
    fields = text.split('\t')
    out = np.empty(len(fields), dtype=np.float32)
    for i, f in enumerate(fields):
        if f == '':
            out[i] = np.nan
            continue
        try:
            out[i] = np.float32(f)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f'Feature column {column!r} position {i} is not numeric: {f!r}'
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
        raw = _to_float_array(_get(row, self.features_column), self.features_column)

        # The cd file references positions inside the feature vector; a row
        # shorter than expected means the cd file and the table are out of sync.
        if raw.shape[0] <= spec.max_feature_pos:
            raise ValueError(
                f'Row has {raw.shape[0]} values in column {self.features_column!r}, '
                f'but cd.txt references feature position {spec.max_feature_pos}. '
                f'The cd file and the table schema are out of sync.'
            )

        features = raw[spec.feature_positions]
        label = float(raw[spec.label_pos])
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
