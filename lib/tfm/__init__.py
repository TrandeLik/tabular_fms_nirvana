import torch

from lib.tfm.base import TFMBase

# NOTE: the wrapper is imported lazily inside `load_tfm` so its (heavy) deps are
# only pulled in when the model is actually requested.


def load_tfm(
    name: str,
    device: torch.device,
    config: dict = {},
) -> TFMBase:
    match name:
        case "tabicl":
            from lib.tfm.tabicl_wrapper import TabICLWrapper

            return TabICLWrapper(device=device, **config)
        case _:
            raise ValueError(f"Unknown tfm: {name}")
