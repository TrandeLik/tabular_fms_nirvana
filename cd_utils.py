"""Parsing of CatBoost column-description (``cd``) files.

A ``cd`` file assigns a role to each *CatBoost dataset column*, numbered from 0
as they appear in the original dataset (see
https://catboost.ai/docs/en/concepts/input-data_column-descfile). In this
project the values are delivered by a YT table split into two places:

* the **id** column (``SampleId`` / ``DocId``) is a *separate* YT column,
  addressed by the name in the cd line's third field. It is not part of the
  feature list. It is **required**.
* every remaining column lives inside a single **list-valued** YT column (its
  name is configurable, default ``features``), with ``Label`` as its first
  element.

Service columns (``SampleId``, ``Weight``, ``Auxiliary`` …) precede ``Label`` in
the cd file and are not part of the feature list, so a column at ``cd`` index
``i`` maps to feature-list position ``i - label_index``. Example::

    0\tSampleId\tkey                          # separate YT column 'key'
    1\tLabel\tis_fraud                         # -> features[0]
    2\tNum\tFActiveDayPercent(Record)          # -> features[1]
    3\tNum\tFBaseAntifraudAction1d(DeviceId)   # -> features[2]

``Label`` is **required** and is the first element of the feature list. Only
``Num`` and ``Categ`` columns become model features (in ``cd`` order). Any other
type after ``Label`` is excluded from the feature matrix; the parser never
raises on an unrecognized *feature* type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# Recognized role tokens (compared case-insensitively).
_LABEL = {'label'}
_DOCID = {'docid', 'sampleid'}
_WEIGHT = {'weight'}
_NUM = {'num'}
_CATEG = {'categ', 'categorical'}


@dataclass
class CdSpec:
    """Parsed column-description.

    Feature/label positions are **positions inside the feature list** (``cd``
    index minus the label's index). The id column is addressed by name via
    :attr:`docid_column`, not by a feature-list position.
    """

    label_pos: int  # position of the target in the feature list (0 = first)
    docid_column: str  # name of the separate YT id column (required)
    num_positions: list[int] = field(default_factory=list)
    cat_positions: list[int] = field(default_factory=list)
    # Feature positions in cd order (num + cat interleaved as declared).
    feature_positions: list[int] = field(default_factory=list)
    # Human-readable names aligned with ``feature_positions``.
    feature_names: list[str] = field(default_factory=list)
    # True for each entry of ``feature_positions`` iff it is categorical.
    feature_is_cat: list[bool] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.feature_positions)

    @property
    def max_feature_pos(self) -> int:
        """Largest feature-list position referenced (features or label)."""
        return max([self.label_pos, *self.feature_positions])


def parse_cd(path: str | Path) -> CdSpec:
    """Parse a CatBoost ``cd`` file into a :class:`CdSpec`.

    Blank lines and ``#`` comments are skipped. Fields are tab-separated
    (surrounding whitespace tolerated).

    Raises ``ValueError`` on: malformed/negative/duplicate indices; a missing or
    unnamed ``SampleId``/``DocId``; a missing ``Label``; an id column that does
    not precede ``Label``; a feature column that precedes ``Label``; or no
    ``Num``/``Categ`` feature columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Column-description file not found: {path}')

    # >>> First pass: collect entries and locate the id + label columns.
    entries: list[tuple[int, str, str, int]] = []  # (index, type, name, lineno)
    label_index: int | None = None
    docid_index: int | None = None
    docid_column: str | None = None
    seen_indices: dict[int, int] = {}  # index -> line number that claimed it

    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue

        fields = [f.strip() for f in line.split('\t') if f.strip() != '']
        if len(fields) < 2:
            raise ValueError(
                f'{path}:{lineno}: expected at least "<index>\\t<type>", '
                f'got {raw_line!r}'
            )

        try:
            index = int(fields[0])
        except ValueError as err:
            raise ValueError(
                f'{path}:{lineno}: column index must be an integer, '
                f'got {fields[0]!r}'
            ) from err
        if index < 0:
            raise ValueError(
                f'{path}:{lineno}: column index must be non-negative, got {index}'
            )
        if index in seen_indices:
            raise ValueError(
                f'{path}:{lineno}: column index {index} is assigned a role more '
                f'than once (first at line {seen_indices[index]})'
            )
        seen_indices[index] = lineno

        col_type = fields[1].lower()
        name = fields[2] if len(fields) >= 3 else ''

        if col_type in _DOCID:
            if docid_index is not None:
                raise ValueError(f'{path}:{lineno}: duplicate SampleId/DocId column')
            if not name:
                raise ValueError(
                    f'{path}:{lineno}: SampleId/DocId must name its YT column, '
                    f'e.g. "{index}\\tSampleId\\t<column_name>"'
                )
            docid_index, docid_column = index, name
        elif col_type in _LABEL:
            if label_index is not None:
                raise ValueError(f'{path}:{lineno}: duplicate Label column')
            label_index = index
        else:
            entries.append((index, col_type, name, lineno))

    # >>> Required columns.
    if docid_column is None or docid_index is None:
        raise ValueError(
            f'{path}: SampleId/DocId column is required (it is the row key) and '
            f'must name its YT column.'
        )
    if label_index is None:
        raise ValueError(
            f'{path}: Label column is required and must be the first element of '
            f'the feature list.'
        )
    if docid_index > label_index:
        raise ValueError(
            f'{path}: SampleId/DocId (index {docid_index}) must precede Label '
            f'(index {label_index}); service columns come before Label.'
        )

    # >>> Second pass: map feature columns to feature-list positions. Positions
    # are measured from Label (Label is feature-list position 0), because every
    # column before Label is a separate/service column absent from the list.
    spec = CdSpec(label_pos=0, docid_column=docid_column)
    for index, col_type, name, lineno in entries:
        pos = index - label_index
        if col_type in _WEIGHT:
            logger.debug(f'{path}:{lineno}: Weight column parsed but ignored')
            continue
        if col_type in _NUM or col_type in _CATEG:
            if pos < 1:
                raise ValueError(
                    f'{path}:{lineno}: feature column at index {index} precedes '
                    f'Label (index {label_index}); Label must be the first '
                    f'element of the feature list.'
                )
            is_cat = col_type in _CATEG
            (spec.cat_positions if is_cat else spec.num_positions).append(pos)
            spec.feature_positions.append(pos)
            spec.feature_names.append(name)
            spec.feature_is_cat.append(is_cat)
        else:
            logger.debug(
                f'{path}:{lineno}: excluding column {index} '
                f'with unhandled type {name or col_type!r}'
            )

    if not spec.feature_positions:
        raise ValueError(
            f'{path}: no model features declared; the cd file must contain at '
            f'least one Num or Categ column'
        )

    logger.info(
        f'Parsed {path.name}: {spec.n_features} model features '
        f'({len(spec.num_positions)} num, {len(spec.cat_positions)} cat), '
        f'label_pos={spec.label_pos}, docid_column={spec.docid_column!r}'
    )
    return spec


def select_features(feature_list: list, spec: CdSpec) -> list:
    """Pick the model feature values from the feature list, in cd order."""
    return [feature_list[i] for i in spec.feature_positions]


def extract_label(feature_list: list, spec: CdSpec):
    """Return the label value from the feature list (Label is required)."""
    return feature_list[spec.label_pos]
