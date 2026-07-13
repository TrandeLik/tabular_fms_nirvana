"""In-context-learning inference core for TFMs.

Extracted from ``main.py`` so it can be exercised independently of YT. The TFM
is trivially inductive: ``(x_train, y_train)`` form the ICL context and
predictions on ``x_eval`` are produced from features alone.

Two entry points:

* :func:`predict_batch` -- predict one already-normalized eval batch, with
  ensembling (feature permutation + optional context subsample) and CUDA-OOM
  chunk-size halving. Returns probabilities (classification) or values
  (regression). This is what the streaming loop calls per batch.
* :class:`ContextEnsemble` -- precomputes the per-member context permutations/
  subsamples once, so repeated ``predict_batch`` calls over a stream reuse them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split

import lib
from lib.tfm.base import TFMBase
from lib.util import TaskType


@dataclass
class _Member:
    perm: torch.Tensor  # feature permutation
    x_train: torch.Tensor  # (permuted) context features for this member
    y_train: torch.Tensor


class ContextEnsemble:
    """Fixed ICL context prepared for ``n_ensemble`` members.

    Each member uses its own feature permutation. When ``max_context_size`` is
    smaller than the context, each member also draws an independent context
    subsample (stratified for classification).
    """

    def __init__(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        *,
        task_type: TaskType,
        n_ensemble: int = 1,
        max_context_size: int | None = None,
        seed: int = 0,
    ) -> None:
        assert n_ensemble >= 1
        self.task_type = task_type
        self.device = x_context.device
        n_context, n_features = x_context.shape
        self.n_features = n_features

        resample = max_context_size is not None and n_context > max_context_size
        y_np = y_context.cpu().numpy()

        self.members: list[_Member] = []
        for m in range(n_ensemble):
            member_seed = seed + m
            rng = np.random.default_rng(member_seed)

            if resample:
                if task_type in (TaskType.BINCLASS, TaskType.MULTICLASS):
                    keep_idx, _ = train_test_split(
                        np.arange(n_context),
                        train_size=max_context_size,
                        stratify=y_np,
                        random_state=member_seed,
                    )
                else:
                    keep_idx = rng.choice(
                        n_context, size=max_context_size, replace=False
                    )
                keep = torch.from_numpy(np.sort(keep_idx)).to(self.device)
                x_m = x_context[keep]
                y_m = y_context[keep]
            else:
                x_m = x_context
                y_m = y_context

            perm = torch.from_numpy(rng.permutation(n_features)).to(self.device)
            self.members.append(_Member(perm=perm, x_train=x_m[:, perm], y_train=y_m))

        self.context_size = self.members[0].x_train.shape[0]
        logger.info(
            f'Context ensemble ready: n_features={n_features} '
            f'context_size={self.context_size} n_ensemble={n_ensemble} '
            f'resample={resample}'
        )


def _forward_chunked(
    model: TFMBase,
    member: _Member,
    x_eval: torch.Tensor,
    task_type: TaskType,
    eval_chunk_size: int,
) -> tuple[torch.Tensor, int]:
    """Forward one member over ``x_eval`` in chunks, halving on CUDA OOM.

    Returns the raw model output (log-probs for classification, values for
    regression) and the (possibly reduced) chunk size to carry forward.
    """
    n_eval = x_eval.shape[0]
    x_eval_p = x_eval[:, member.perm]
    out_chunks: list[torch.Tensor] = []
    start = 0
    while start < n_eval:
        chunk = x_eval_p[start : start + eval_chunk_size]
        try:
            with torch.inference_mode():
                out_chunks.append(
                    model.forward(
                        x_train=member.x_train,
                        y_train=member.y_train,
                        x_eval=chunk,
                        task_type=task_type,
                    )
                )
            start += chunk.shape[0]
        except RuntimeError as err:
            if not lib.is_oom_exception(err) or eval_chunk_size <= 1:
                raise
            new_size = max(1, eval_chunk_size // 2)
            logger.warning(
                f'OOM at eval_chunk_size={eval_chunk_size}, retrying with {new_size}'
            )
            eval_chunk_size = new_size
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    return torch.cat(out_chunks, dim=0).float(), eval_chunk_size


def predict_batch(
    model: TFMBase,
    ensemble: ContextEnsemble,
    x_eval: torch.Tensor,
    *,
    eval_chunk_size: int | None = None,
    _chunk_state: list[int] | None = None,
) -> np.ndarray:
    """Predict one eval batch, averaging ensemble members in probability space.

    Args:
        model: the TFM wrapper.
        ensemble: prepared :class:`ContextEnsemble`.
        x_eval: normalized eval features, shape (n_eval, n_features).
        eval_chunk_size: initial chunk size (defaults to the whole batch).
        _chunk_state: optional single-element list holding the adapted chunk
            size, so OOM-driven shrinking persists across batches.

    Returns:
        Classification: probabilities, shape (n_eval, n_classes).
        Regression: values, shape (n_eval,).
    """
    task_type = ensemble.task_type
    n_eval = x_eval.shape[0]
    initial = (
        _chunk_state[0]
        if _chunk_state
        else (eval_chunk_size if eval_chunk_size is not None else n_eval)
    )
    # Start this batch at the persistent size, capped by the batch itself.
    start_chunk = min(initial, n_eval) if n_eval else initial

    accum: torch.Tensor | None = None
    chunk = start_chunk
    for member in ensemble.members:
        out, chunk = _forward_chunked(model, member, x_eval, task_type, chunk)
        if task_type in (TaskType.BINCLASS, TaskType.MULTICLASS):
            out = out.exp()  # wrapper returns log-probs
        accum = out if accum is None else accum + out

    assert accum is not None
    averaged = (accum / len(ensemble.members)).cpu().numpy()
    if _chunk_state is not None:
        # Only a genuine OOM (chunk shrank below the batch-capped start) should
        # lower the persistent size; a small batch must not penalize later ones.
        _chunk_state[0] = chunk if chunk < start_chunk else initial
    return averaged
