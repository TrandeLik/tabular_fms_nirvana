"""Parsing of CatBoost column-description (``cd``) files.

A ``cd`` file assigns a role to each position of the model's feature vector.
Unlike the classic CatBoost layout (where the index refers to a positional
column of a CSV), here every index refers to a position *within a single
list-valued column* of a YT table (its name is configurable, default
``features``). Example::

    0\tAuxiliary
    1\tLabel
    2\tDocId
    3\tWeight
    4\tNum\tTRACK_TIME_DAY_TYPE
    5\tNum\tTRACK_TIME_MINUTE

Only ``Num`` and ``Categ`` positions become model features (in ``cd`` order).
``Label``/``DocId`` are extracted as the target and the row key. ``Weight`` is
recognized but currently ignored. Every other/unknown type -- and every
position without a ``cd`` line -- is excluded from the feature matrix. The
parser is intentionally permissive and never raises on an unrecognized type.
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

    Indices refer to positions inside the list-valued feature column.
    """

    label_idx: int | None = None
    docid_idx: int | None = None
    weight_idx: int | None = None
    num_indices: list[int] = field(default_factory=list)
    cat_indices: list[int] = field(default_factory=list)
    # Feature positions in cd order (num + cat interleaved as declared).
    feature_indices: list[int] = field(default_factory=list)
    # Human-readable names aligned with ``feature_indices`` (empty string when
    # the cd line omits a name).
    feature_names: list[str] = field(default_factory=list)
    # True for each entry of ``feature_indices`` iff it is categorical.
    feature_is_cat: list[bool] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.feature_indices)

    @property
    def has_label(self) -> bool:
        return self.label_idx is not None

    @property
    def max_index(self) -> int:
        """Largest position referenced by any role (features/label/docid/weight)."""
        indices = list(self.feature_indices)
        for idx in (self.label_idx, self.docid_idx, self.weight_idx):
            if idx is not None:
                indices.append(idx)
        return max(indices) if indices else -1


def parse_cd(path: str | Path) -> CdSpec:
    """Parse a CatBoost ``cd`` file into a :class:`CdSpec`.

    Blank lines and ``#`` comments are skipped. Fields are tab-separated
    (surrounding whitespace tolerated). Unknown types are logged and excluded.
    Raises on malformed indices or duplicate ``Label``/``DocId`` declarations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Column-description file not found: {path}')

    spec = CdSpec()
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

        if col_type in _LABEL:
            if spec.label_idx is not None:
                raise ValueError(f'{path}:{lineno}: duplicate Label column')
            spec.label_idx = index
        elif col_type in _DOCID:
            if spec.docid_idx is not None:
                raise ValueError(f'{path}:{lineno}: duplicate DocId/SampleId column')
            spec.docid_idx = index
        elif col_type in _WEIGHT:
            if spec.weight_idx is not None:
                raise ValueError(f'{path}:{lineno}: duplicate Weight column')
            spec.weight_idx = index  # parsed but currently unused
        elif col_type in _NUM:
            spec.num_indices.append(index)
            spec.feature_indices.append(index)
            spec.feature_names.append(name)
            spec.feature_is_cat.append(False)
        elif col_type in _CATEG:
            spec.cat_indices.append(index)
            spec.feature_indices.append(index)
            spec.feature_names.append(name)
            spec.feature_is_cat.append(True)
        else:
            logger.debug(
                f'{path}:{lineno}: excluding column {index} '
                f'with unhandled type {fields[1]!r}'
            )

    if not spec.feature_indices:
        raise ValueError(
            f'{path}: no model features declared; the cd file must contain at '
            f'least one Num or Categ column'
        )

    logger.info(
        f'Parsed {path.name}: {spec.n_features} model features '
        f'({len(spec.num_indices)} num, {len(spec.cat_indices)} cat), '
        f'label={"yes" if spec.has_label else "no"}, '
        f'docid={"yes" if spec.docid_idx is not None else "no"}'
    )
    return spec


def select_features(row_features: list, spec: CdSpec) -> list:
    """Pick the model feature values from a full feature list, in cd order."""
    return [row_features[i] for i in spec.feature_indices]


def extract_label(row_features: list, spec: CdSpec):
    """Return the label value or ``None`` when the cd declares no Label."""
    if spec.label_idx is None:
        return None
    return row_features[spec.label_idx]


def extract_docid(row_features: list, spec: CdSpec):
    """Return the row-key value or ``None`` when the cd declares no DocId."""
    if spec.docid_idx is None:
        return None
    return row_features[spec.docid_idx]
