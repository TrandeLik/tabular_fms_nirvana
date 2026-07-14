"""Feature normalization fit on the ICL context and reused per streamed batch.

A small, sklearn-only implementation of the numerical/categorical transforms
needed for tabular TFM inference (no graph/dgl dependencies).

Transforms are fit once on the downloaded context and then applied unchanged to
every streamed test batch, so context and test are normalized consistently. The
feature matrix columns are in ``cd`` order (as produced by
:func:`cd_utils.select_features`); numerical and categorical columns are
transformed in place and reassembled into the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sklearn.impute
import sklearn.preprocessing

from cd_utils import CdSpec
from config import Config

# num_policy values.
_STANDARD = 'standard'
_QUANTILE_NORMAL = 'quantile-normal'
_QUANTILE_UNIFORM = 'quantile-uniform'
# cat_policy values.
_CAT_ORDINAL = 'ordinal'
_CAT_STANDARD = 'standard'
_CAT_ONE_HOT = 'one-hot'
# impute_strategy values.
_IMPUTE_BASIC = 'basic'
_IMPUTE_STANDARDIZE_MIN = 'standardize_min'


@dataclass
class FittedTransformers:
    """Fitted state + column layout needed to transform new batches."""

    n_input_features: int  # expected width of the raw (cd-ordered) feature matrix
    num_cols: np.ndarray  # positions (within the cd-ordered feature matrix)
    cat_cols: np.ndarray
    num_transformer: object | None
    num_imputer: object | None
    num_fill: np.ndarray | None  # per-column fill for 'standardize_min'
    cat_encoder: object | None
    cat_scaler: object | None


def _build_num_transformer(policy: str | None, seed: int):
    if policy is None:
        return None
    if policy == _STANDARD:
        return sklearn.preprocessing.StandardScaler()
    if policy in (_QUANTILE_NORMAL, _QUANTILE_UNIFORM):
        distribution = 'normal' if policy == _QUANTILE_NORMAL else 'uniform'
        return sklearn.preprocessing.QuantileTransformer(
            output_distribution=distribution,
            subsample=1_000_000_000,
            random_state=seed,
        )
    raise ValueError(f'Unknown num_policy: {policy!r}')


def fit_transform_context(
    raw_features: np.ndarray,
    cd_spec: CdSpec,
    config: Config,
) -> tuple[np.ndarray, FittedTransformers]:
    """Fit transformers on the context features and return the transformed matrix.

    Args:
        raw_features: context feature matrix in cd order, shape (n_context, n_feat).
        cd_spec: parsed column-description (provides per-column num/cat flags).
        config: run config (normalization policies + seed).
    """
    raw_features = np.asarray(raw_features, dtype=np.float32)
    if raw_features.ndim != 2:
        raise ValueError(
            f'context features must be a 2D array, got shape {raw_features.shape}'
        )
    if raw_features.shape[0] == 0:
        raise ValueError('context is empty; cannot fit feature normalization')
    if raw_features.shape[1] != cd_spec.n_features:
        raise ValueError(
            f'context has {raw_features.shape[1]} feature columns but cd.txt '
            f'declares {cd_spec.n_features}'
        )

    is_cat = np.asarray(cd_spec.feature_is_cat, dtype=bool)
    num_cols = np.flatnonzero(~is_cat)
    cat_cols = np.flatnonzero(is_cat)

    fitted = FittedTransformers(
        n_input_features=raw_features.shape[1],
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_transformer=None,
        num_imputer=None,
        num_fill=None,
        cat_encoder=None,
        cat_scaler=None,
    )

    blocks: list[tuple[np.ndarray, np.ndarray]] = []  # (column positions, values)

    # >>> Numerical block
    if num_cols.size:
        num = raw_features[:, num_cols]
        num, fitted.num_transformer, fitted.num_imputer, fitted.num_fill = (
            _fit_num(num, config)
        )
        blocks.append((num_cols, num))

    # >>> Categorical block
    if cat_cols.size:
        cat = raw_features[:, cat_cols]
        cat, fitted.cat_encoder, fitted.cat_scaler = _fit_cat(cat, config)
        blocks.append((cat_cols, cat))

    transformed = _reassemble(raw_features.shape[0], num_cols, cat_cols, blocks, fitted)
    return transformed, fitted


def transform_batch(
    raw_features: np.ndarray,
    fitted: FittedTransformers,
) -> np.ndarray:
    """Apply fitted transformers to a streamed batch (cd-ordered features)."""
    raw_features = np.asarray(raw_features, dtype=np.float32)
    if raw_features.ndim != 2:
        raise ValueError(
            f'batch features must be a 2D array, got shape {raw_features.shape}'
        )
    if raw_features.shape[1] != fitted.n_input_features:
        raise ValueError(
            f'batch has {raw_features.shape[1]} feature columns but the fitted '
            f'pipeline expects {fitted.n_input_features}'
        )
    blocks: list[tuple[np.ndarray, np.ndarray]] = []

    if fitted.num_cols.size:
        num = raw_features[:, fitted.num_cols]
        blocks.append((fitted.num_cols, _apply_num(num, fitted)))
    if fitted.cat_cols.size:
        cat = raw_features[:, fitted.cat_cols]
        blocks.append((fitted.cat_cols, _apply_cat(cat, fitted)))

    return _reassemble(
        raw_features.shape[0], fitted.num_cols, fitted.cat_cols, blocks, fitted
    )


# ==================================================================================
# Numerical
# ==================================================================================
def _fit_num(num: np.ndarray, config: Config):
    transformer = _build_num_transformer(config.num_policy, config.seed)
    imputer = None
    fill = None
    strategy = config.impute_strategy

    if transformer is not None:
        transformer.fit(num)
        num = transformer.transform(num)

    if strategy == _IMPUTE_STANDARDIZE_MIN:
        # Fill NaNs with each column's minimal (seen) value.
        fill = np.nanmin(num, axis=0)
        num = _fill_nan(num, fill)
    elif strategy == _IMPUTE_BASIC:
        imputer = sklearn.impute.SimpleImputer()
        imputer.fit(num)
        num = imputer.transform(num)
    elif strategy is not None:
        raise ValueError(f'Unknown impute_strategy: {strategy!r}')

    return num.astype(np.float32), transformer, imputer, fill


def _apply_num(num: np.ndarray, fitted: FittedTransformers) -> np.ndarray:
    if fitted.num_transformer is not None:
        num = fitted.num_transformer.transform(num)
    if fitted.num_fill is not None:
        num = _fill_nan(num, fitted.num_fill)
    elif fitted.num_imputer is not None:
        num = fitted.num_imputer.transform(num)
    return num.astype(np.float32)


def _fill_nan(x: np.ndarray, fill: np.ndarray) -> np.ndarray:
    x = x.copy()
    nan_idx = np.isnan(x)
    if nan_idx.any():
        x[nan_idx] = np.take(fill, np.nonzero(nan_idx)[1])
    return x


# ==================================================================================
# Categorical
# ==================================================================================
def _fit_cat(cat: np.ndarray, config: Config):
    policy = config.cat_policy
    if policy is None:
        return cat.astype(np.float32), None, None

    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        dtype=np.float32,
    ).fit(cat)
    encoded = encoder.transform(cat)

    scaler = None
    if policy == _CAT_ORDINAL:
        pass
    elif policy == _CAT_STANDARD:
        scaler = sklearn.preprocessing.StandardScaler().fit(encoded)
        encoded = scaler.transform(encoded)
    elif policy == _CAT_ONE_HOT:
        scaler = sklearn.preprocessing.OneHotEncoder(
            drop='if_binary',
            sparse_output=False,
            handle_unknown='ignore',
            dtype=np.float32,
        ).fit(encoded)
        encoded = scaler.transform(encoded)
    else:
        raise ValueError(f'Unknown cat_policy: {policy!r}')

    return encoded.astype(np.float32), encoder, scaler


def _apply_cat(cat: np.ndarray, fitted: FittedTransformers) -> np.ndarray:
    if fitted.cat_encoder is None:
        return cat.astype(np.float32)
    encoded = fitted.cat_encoder.transform(cat)
    if fitted.cat_scaler is not None:
        encoded = fitted.cat_scaler.transform(encoded)
    return encoded.astype(np.float32)


# ==================================================================================
# Reassembly
# ==================================================================================
def _reassemble(
    n_rows: int,
    num_cols: np.ndarray,
    cat_cols: np.ndarray,
    blocks: list[tuple[np.ndarray, np.ndarray]],
    fitted: FittedTransformers,
) -> np.ndarray:
    """Reassemble num/cat blocks into a single matrix.

    When neither policy changes the column count (i.e. no one-hot expansion),
    columns are placed back at their original cd positions so the feature order
    matches ``cd_spec``. If one-hot expansion is used, columns are concatenated
    num-block then cat-block (the exact order is stable across context/test,
    which is all the model needs).
    """
    onehot = (
        fitted.cat_scaler is not None
        and isinstance(fitted.cat_scaler, sklearn.preprocessing.OneHotEncoder)
    )
    if not onehot:
        width = num_cols.size + cat_cols.size
        out = np.empty((n_rows, width), dtype=np.float32)
        for cols, values in blocks:
            out[:, cols] = values
        return out
    return np.concatenate([values for _, values in blocks], axis=1).astype(np.float32)
