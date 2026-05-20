from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import entropy, kurtosis, skew
from transformers import AutoModelForCausalLM, AutoTokenizer


FEATURE_NAMES: Tuple[str, ...] = (
    "mean_surprisal",
    "stdev_surprisal",
    "var_surprisal",
    "skew_surprisal",
    "kurtosis_surprisal",
    "mean_diff_surprisal",
    "stdev_diff_surprisal",
    "var_second_diff_loglik",
    "entropy_second_diff_loglik",
    "autocorr_second_diff_loglik",
)


@dataclass(frozen=True)
class RaidFeatureResult:
    model_name: str
    token_log_likelihoods: int
    features: Dict[str, float]


class RaidFeatureExtractor:
    def __init__(self, model_name: str, *, device: str = "auto", max_length: int = 1024) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be > 0")

        self.model_name = model_name
        self.max_length = max_length

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _token_log_likelihoods(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        if tokens.numel() < 2:
            raise ValueError("Need at least 2 tokens to compute log-likelihoods")

        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].squeeze(0)
        shift_labels = tokens[:, 1:].squeeze(0)
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        token_log_likelihoods = log_probs[range(shift_labels.shape[0]), shift_labels]

        return token_log_likelihoods.cpu().numpy()

    def compute(self, text: str) -> RaidFeatureResult:
        log_likelihoods = self._token_log_likelihoods(text)

        if len(log_likelihoods) < 10:
            raise ValueError("Need at least 10 token log-likelihoods to compute Raid features")

        surprisals = -log_likelihoods
        s = np.array(surprisals, dtype=np.float64)

        mean_s = float(np.mean(s))
        std_s = float(np.std(s))
        var_s = float(np.var(s))
        skew_s = float(skew(s))
        kurt_s = float(kurtosis(s))

        diff_s = np.diff(s)
        mean_diff = float(np.mean(diff_s))
        std_diff = float(np.std(diff_s))

        first_order_diff = np.diff(log_likelihoods)
        second_order_diff = np.diff(first_order_diff)
        var_2nd = float(np.var(second_order_diff))
        entropy_2nd = float(
            entropy(np.histogram(second_order_diff, bins=20, density=True)[0])
        )
        autocorr_2nd = (
            float(np.corrcoef(second_order_diff[:-1], second_order_diff[1:])[0, 1])
            if len(second_order_diff) > 1
            else 0.0
        )

        values = [
            mean_s,
            std_s,
            var_s,
            skew_s,
            kurt_s,
            mean_diff,
            std_diff,
            var_2nd,
            entropy_2nd,
            autocorr_2nd,
        ]

        features = dict(zip(FEATURE_NAMES, values))

        return RaidFeatureResult(
            model_name=self.model_name,
            token_log_likelihoods=int(len(log_likelihoods)),
            features=features,
        )


def result_to_json(result: RaidFeatureResult) -> Dict[str, object]:
    return asdict(result)

