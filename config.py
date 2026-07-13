"""Run configuration for TFM inference on YT tables.

The config is a plain dataclass loaded from a YAML file (``CONFIG.yaml``),
following the pattern of ``dev/config.py``: this is the easiest way to expose
hyperparameters as Nirvana operation parameters. Unknown keys in the YAML are
rejected by the dataclass constructor, which keeps typos loud.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    features_column: str = 'features'
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

    # >>> Inference
    seed: int = 0
    # Subsample the context to at most this many rows (None = use all).
    max_context_size: int | None = None
    # Number of ensemble members (feature permutation + context subsample),
    # averaged in probability space.
    n_ensemble: int = 1
    # Rows per forward pass on the eval side (None = whole batch at once).
    eval_chunk_size: int | None = None

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
        config_dict: dict[str, Any] = yaml.safe_load(f_read) or {}

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
