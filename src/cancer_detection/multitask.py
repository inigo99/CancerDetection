"""Joint prediction: one backbone, three heads (age, cancer, density)."""

import math

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

AGE_SCALE = math.log(89)  # the oldest patient: log(age) / AGE_SCALE falls in (0, 1]


def encode_age(age: torch.Tensor) -> torch.Tensor:
    return age.float().log() / AGE_SCALE


def decode_age(y: torch.Tensor) -> torch.Tensor:
    return (y * AGE_SCALE).exp()


class MultiTaskModel(nn.Module):
    """Any encoder that returns a (batch, n_features) tensor, plus one head per task."""

    def __init__(self, encoder: nn.Module, n_features: int, p: float = 0.25):
        super().__init__()
        self.encoder = encoder

        def head(n_out):
            return nn.Sequential(nn.Dropout(p), nn.Linear(n_features, n_out))

        self.age, self.cancer, self.density = head(1), head(2), head(4)

    def forward(self, x):
        z = self.encoder(x)
        return torch.sigmoid(self.age(z)).squeeze(-1), self.cancer(z), self.density(z)


class UncertaintyWeightedLoss(nn.Module):
    """Kendall et al. uncertainty weighting on top of fixed task weights (age, cancer, density).

    The thesis's best run used weights 0.1 / 0.6 / 0.3."""

    def __init__(self, weights=(0.1, 0.6, 0.3)):
        super().__init__()
        self.weights = weights
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.mse, self.ce = nn.MSELoss(), nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        losses = (
            self.mse(preds[0], targets[0]),
            self.ce(preds[1], targets[1]),
            self.ce(preds[2], targets[2]),
        )
        return 3 * sum(
            w * (torch.exp(-lv) * loss + lv)
            for w, lv, loss in zip(self.weights, self.log_vars, losses)
        )


def f1_cancer(cancer_logits: torch.Tensor, labels: torch.Tensor) -> float:
    """F1 for the cancer head. Class 1 (cancer) is predicted when its logit is the larger one;
    the notebook's version predicted 1 when logit 0 was larger, i.e. the labels were flipped."""
    pred = cancer_logits.argmax(dim=1).cpu().numpy()
    return float(f1_score(np.asarray(labels.cpu()), pred, zero_division=0))
