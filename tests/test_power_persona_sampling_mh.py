import math

import pytest

torch = pytest.importorskip("torch")

from power_persona_sampling.config import PersonaSpec, SamplerConfig
from power_persona_sampling.lm.base import BaseLM, BatchScoreRequest, BatchScoreResult, SamplingParams
from power_persona_sampling.mh import log_accept_ratio_suffix_resample
from power_persona_sampling.persona import PersonaVector, PersonaVectorSet
from power_persona_sampling.sampler import MHSampler


class FakeLM(BaseLM):
    def __init__(self, sample_suffix_tokens: list[int]):
        self._sample_suffix_tokens = list(sample_suffix_tokens)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        text = text.strip()
        if not text:
            return []
        return [int(tok) for tok in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def sample_suffix(self, prefix_ids, params: SamplingParams) -> list[int]:
        return list(self._sample_suffix_tokens)

    def score_batch(self, req: BatchScoreRequest) -> BatchScoreResult:
        logp_response = []
        logp_suffix = []
        pooled_by_layer = {}

        for b in range(len(req.prompt_lens)):
            prompt_len = req.prompt_lens[b]
            response_len = req.response_lens[b]
            response_ids = req.input_ids[b, prompt_len : prompt_len + response_len].tolist()
            logp_response.append(-float(sum(response_ids)))
            suffix_start = req.suffix_start[b] if req.suffix_start is not None else 0
            logp_suffix.append(-float(sum(response_ids[suffix_start:])))

        for layer in req.layers:
            pooled = []
            for b in range(len(req.prompt_lens)):
                prompt_len = req.prompt_lens[b]
                response_len = req.response_lens[b]
                response_ids = req.input_ids[b, prompt_len : prompt_len + response_len].tolist()
                if response_len == 0:
                    pooled.append(torch.zeros(1))
                else:
                    pooled.append(torch.tensor([sum(response_ids) / response_len], dtype=torch.float))
            pooled_by_layer[layer] = torch.stack(pooled, dim=0)

        return BatchScoreResult(
            logp_response=logp_response,
            logp_suffix=logp_suffix,
            pooled_by_layer=pooled_by_layer,
        )


def test_log_accept_ratio_suffix_resample_combines_terms():
    log_r = log_accept_ratio_suffix_resample(
        alpha=2.0,
        logp_suffix_x=-5.0,
        logp_suffix_xp=-2.0,
        persona_x={"a": 1.0, "b": -1.0},
        persona_xp={"a": 1.5, "b": -2.0},
        persona_weights={"a": 2.0, "b": 0.5},
    )

    expected_delta_logp = -2.0 - (-5.0)
    expected_delta_persona = 2.0 * (1.5 - 1.0) + 0.5 * (-2.0 - (-1.0))
    expected = (2.0 - 1.0) * expected_delta_logp + expected_delta_persona
    assert math.isclose(log_r, expected)


def test_mh_sampler_accepts_persona_gain():
    lm = FakeLM(sample_suffix_tokens=[9, 0])
    vectors = PersonaVectorSet([PersonaVector(name="trait", layer=0, vector=torch.tensor([1.0]))])
    spec = PersonaSpec(name="trait", layer=0, beta=1.0, lam=1.0)
    cfg = SamplerConfig(alpha=1.0, max_new_tokens=3, seed=1, burn_in=0, thin=1)
    sampler = MHSampler(lm, vectors, [spec], cfg)

    result = sampler.sample(prompt="5", num_steps=1, init_response="1 1", show_progress=False)

    assert result.stats[0].accepted is True
    assert result.final.traj.response_ids == [9, 0]
    assert len(result.kept) == 1


def test_mh_sampler_rejects_persona_drop():
    lm = FakeLM(sample_suffix_tokens=[0])
    vectors = PersonaVectorSet([PersonaVector(name="trait", layer=0, vector=torch.tensor([1.0]))])
    spec = PersonaSpec(name="trait", layer=0, beta=1000.0, lam=1.0)
    cfg = SamplerConfig(alpha=1.0, max_new_tokens=3, seed=1, burn_in=0, thin=1)
    sampler = MHSampler(lm, vectors, [spec], cfg)

    result = sampler.sample(prompt="5", num_steps=1, init_response="9 9", show_progress=False)

    assert result.stats[0].accepted is False
    assert result.final.traj.response_ids == [9, 9, 0]
    assert len(result.kept) == 1
