"""Run configuration for TFM inference on YT tables.

The config is a plain dataclass loaded from a YAML file (``CONFIG.yaml``),
following the pattern of ``dev/config.py``: this is the easiest way to expose
hyperparameters as Nirvana operation parameters. Unknown keys in the YAML are
rejected by the dataclass constructor, which keeps typos loud.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Allowed values for the string-enum config fields, mirrored from the code that
# consumes them (lib.util.TaskType, feature_pipeline policies, load_tfm names).
_TASK_TYPES = frozenset({'regression', 'binclass', 'multiclass'})
_NUM_POLICIES = frozenset({'standard', 'quantile-normal', 'quantile-uniform'})
_CAT_POLICIES = frozenset({'ordinal', 'standard', 'one-hot'})
_IMPUTE_STRATEGIES = frozenset({'basic', 'standardize_min'})
_TFM_NAMES = frozenset({'tabicl'})


@dataclass
class Config:
    # >>> Model
    # Name understood by lib.tfm.load_tfm (e.g. 'tabicl'). Extra kwargs for the
    # wrapper go into tfm_config.
    tfm_name: str = 'tabicl'
    tfm_config: dict[str, Any] = field(default_factory=dict)
    # One of 'regression' | 'binclass' | 'multiclass' (lib.util.TaskType values).
    task_type: str = 'binclass'

    # >>> Data
    # Name of the list-valued YT column that holds the raw feature vector; cd.txt
    # indices refer to positions within it.
    features_column: str = 'value'
    batch_size: int = 1024
    # Output table root for temporary result tables on YT.
    output_table_tmp_path: str = '//home/yr/trandelik/crypta/datasets/'

    # >>> Feature normalization (fit on the downloaded context, reused per batch)
    # Numerical policy: None | 'standard' | 'quantile-normal' | 'quantile-uniform'.
    num_policy: str | None = 'standard'
    # Categorical policy: None | 'ordinal' | 'standard' | 'one-hot'.
    cat_policy: str | None = 'ordinal'
    # Imputation for numerical NaNs: None | 'basic' | 'standardize_min'.
    impute_strategy: str | None = 'basic'
    # Whether feature normalization is fit before or after target filtering.
    # If True, statistics are fit on ALL downloaded context rows (including
    # rows later dropped for having no target); if False (default), they are
    # fit only on the target-labeled rows. Either way the ICL context keeps
    # labeled rows only.
    preprocess_before_target_filter: bool = False

    # >>> Inference
    seed: int = 0
    # Subsample the context to at most this many rows (None = use all).
    max_context_size: int | None = None
    # Number of ensemble members (feature permutation + context subsample),
    # averaged in probability space.
    n_ensemble: int = 1
    # Rows per forward pass on the eval side (None = whole batch at once).
    eval_chunk_size: int | None = None

    def __post_init__(self) -> None:
        """Validate field values so misconfigurations fail fast and clearly."""

        def _one_of(name: str, value, allowed: frozenset[str]) -> None:
            if value not in allowed:
                raise ValueError(
                    f'config.{name}={value!r} is invalid; '
                    f'expected one of {sorted(allowed)}'
                )

        def _optional_one_of(name: str, value, allowed: frozenset[str]) -> None:
            if value is not None:
                _one_of(name, value, allowed)

        def _positive_int(name: str, value) -> None:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f'config.{name}={value!r} must be a positive integer'
                )

        _one_of('tfm_name', self.tfm_name, _TFM_NAMES)
        _one_of('task_type', self.task_type, _TASK_TYPES)
        _optional_one_of('num_policy', self.num_policy, _NUM_POLICIES)
        _optional_one_of('cat_policy', self.cat_policy, _CAT_POLICIES)
        _optional_one_of('impute_strategy', self.impute_strategy, _IMPUTE_STRATEGIES)

        if not isinstance(self.tfm_config, dict):
            raise TypeError(
                f'config.tfm_config must be a mapping, got {type(self.tfm_config).__name__}'
            )
        if not isinstance(self.features_column, str) or not self.features_column:
            raise ValueError('config.features_column must be a non-empty string')
        if not isinstance(self.output_table_tmp_path, str) or not (
            self.output_table_tmp_path.startswith('//')
        ):
            raise ValueError(
                'config.output_table_tmp_path must be an absolute YT path '
                f"(starting with '//'), got {self.output_table_tmp_path!r}"
            )

        _positive_int('batch_size', self.batch_size)
        _positive_int('n_ensemble', self.n_ensemble)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f'config.seed must be an integer, got {self.seed!r}')
        if not isinstance(self.preprocess_before_target_filter, bool):
            raise TypeError(
                'config.preprocess_before_target_filter must be a bool, got '
                f'{type(self.preprocess_before_target_filter).__name__}'
            )
        if self.max_context_size is not None:
            _positive_int('max_context_size', self.max_context_size)
        if self.eval_chunk_size is not None:
            _positive_int('eval_chunk_size', self.eval_chunk_size)

    def get_experiment_name(self) -> str:
        return f'{self.tfm_name}_{self.task_type}'

    def to_dict(self) -> dict[str, Any]:
        def is_param(param_name: str) -> bool:
            return not param_name.startswith('_') and param_name not in {
                'to_dict',
                'get_experiment_name',
            }

        return {name: getattr(self, name) for name in filter(is_param, dir(self))}


def get_config(config_file: Path, config_class: type[Config] = Config) -> Config:
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f'No config found at {config_file}')

    import yaml

    with config_file.open() as f_read:
        try:
            loaded = yaml.safe_load(f_read)
        except yaml.YAMLError as err:
            raise ValueError(f'Failed to parse YAML config {config_file}: {err}') from err

    config_dict: Any = loaded or {}
    if not isinstance(config_dict, dict):
        raise ValueError(
            f'Config {config_file} must contain a mapping at the top level, '
            f'got {type(config_dict).__name__}'
        )

    # Surface unknown keys with a helpful message instead of a bare TypeError.
    known = {f.name for f in fields(config_class)}
    unknown = set(config_dict) - known
    if unknown:
        raise ValueError(
            f'Unknown config keys in {config_file}: {sorted(unknown)}. '
            f'Valid keys are: {sorted(known)}'
        )

    return config_class(**config_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TFM inference on YT tables')
    parser.add_argument('--proxy', default='hahn', help='YT proxy/cluster')
    parser.add_argument(
        '--device',
        default=None,
        help="'cpu', a CUDA index like '0', or None for auto-detect",
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Skip uploading results to YT',
    )
    return parser.parse_args()
