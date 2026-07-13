from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor

from lib.tfm import TFMBase
from lib.tfm.tabicl.model.inference_config import InferenceConfig
from lib.tfm.tabicl.model.tabicl import TabICL
from lib.util import TaskType


def _load_model(
    name: str,
    repo_id: str = 'jingang/TabICL',
    cache_path: str | Path = 'checkpoints',
) -> TabICL:
    model_path = Path(cache_path) / name
    if not model_path.exists():
        Path(hf_hub_download(repo_id, name, local_dir=cache_path)).rename(model_path)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model = TabICL(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


# Adopted from TabICLv2 code
def _standardize_features(x: Tensor) -> Tensor:
    mean = x.mean(dim=0)
    scale = x.std(dim=0) + 1e-6
    x = (x - mean) / scale
    return torch.clip(x, -100.0, 100.0)


# Adopted from TabICLv2 code
def _masked_mean_std(x: Tensor, mask: Tensor, ddof: int) -> tuple[Tensor, Tensor]:
    # Scalar mean/std over masked entries, computed without boolean indexing
    # to avoid CUDA `nonzero` overflow when x has more than INT_MAX elements.
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    n = mask.sum().to(x.dtype).clamp(min=1)
    mean = torch.where(mask, x, zero).sum() / n
    diff_sq = torch.where(mask, (x - mean) ** 2, zero)
    var = diff_sq.sum() / (n - ddof).clamp(min=1)
    return mean, var.sqrt().clamp(min=1e-6)


def _clip_feature_outliers(
    x: Tensor,
    threshold: float = 4.0,
) -> Tensor:
    ddof = 1 if x.shape[0] > 1 else 0

    # Initial statistics
    mask = ~x.isnan()
    means, stds = _masked_mean_std(x, mask, ddof)

    # Recompute without outliers
    outlier_mask = torch.abs(x - means) > threshold * stds
    mask = mask & ~outlier_mask
    means, stds = _masked_mean_std(x, mask, ddof)

    lower = means - threshold * stds
    upper = means + threshold * stds

    # Soft log-based clipping
    log1p = torch.log1p(torch.abs(x))
    x = torch.clip(x, min=-log1p + lower, max=log1p + upper)
    return x


class TabICLWrapper(TFMBase):
    def __init__(
        self,
        device: torch.device,
        classifier_checkpoint: str = 'tabicl-classifier-v2-20260212.ckpt',
        regressor_checkpoint: str = 'tabicl-regressor-v2-20260212.ckpt',
        use_amp: bool | Literal['auto'] = 'auto',
        use_fa3: bool | Literal['auto'] = 'auto',
        softmax_temperature: float = 0.9,
        standardize_features: bool = True,
    ) -> None:
        super().__init__()
        self.classifier = _load_model(classifier_checkpoint)
        self.regressor = _load_model(regressor_checkpoint)
        self._use_amp = use_amp
        self._use_fa3 = use_fa3
        self.softmax_temperature = softmax_temperature
        self.standardize_features = standardize_features
        self._inference_config = InferenceConfig()
        self._inference_config.update_from_dict(
            {
                'COL_CONFIG': {'device': device},
                'ROW_CONFIG': {'device': device},
                'ICL_CONFIG': {'device': device},
            }
        )

    def _resolve_amp_fa3(self, n_samples: int, n_features: int) -> tuple[bool, bool]:
        small_data = n_samples < 1024 and n_features < 60

        if self._use_amp == 'auto':
            use_amp = not small_data
        else:
            use_amp = bool(self._use_amp)

        if self._use_fa3 == 'auto':
            if small_data:
                use_fa3 = False
            elif not use_amp:
                use_fa3 = True
            else:
                use_fa3 = n_samples >= 10240
        else:
            use_fa3 = bool(self._use_fa3)

        return use_amp, use_fa3

    def forward(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_eval: Tensor,
        task_type: TaskType,
    ) -> Tensor:
        x = torch.cat([x_train, x_eval], dim=0)
        if self.standardize_features:
            x = _standardize_features(x)
        x = _clip_feature_outliers(x)
        x = x.unsqueeze(0)

        y_train = y_train.unsqueeze(0)

        use_amp, use_fa3 = self._resolve_amp_fa3(
            n_samples=x.shape[1], n_features=x.shape[2]
        )
        self._inference_config.update_from_dict(
            {
                'COL_CONFIG': {'use_amp': use_amp, 'use_fa3': use_fa3},
                'ROW_CONFIG': {'use_amp': use_amp, 'use_fa3': use_fa3},
                'ICL_CONFIG': {'use_amp': use_amp, 'use_fa3': use_fa3},
            }
        )

        if task_type == TaskType.REGRESSION:
            preds = self.regressor.forward(
                x,
                y_train,
                inference_config=self._inference_config,
            ).float()
            preds = preds.mean(dim=-1)  # NOTE: preds are quantile values (not logits)
        elif task_type in [TaskType.BINCLASS, TaskType.MULTICLASS]:
            preds = self.classifier.forward(
                x,
                y_train,
                inference_config=self._inference_config,
            ).float()
            preds = F.log_softmax(preds / self.softmax_temperature, dim=-1)
        else:
            raise ValueError(f'Unknown {task_type=}')

        return preds.squeeze(0)
