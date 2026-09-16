from types import SimpleNamespace
from pathlib import Path
import runpy

import torch
from torch import nn
from transformers import CLIPTextConfig, CLIPTextModel
from diffusers import DDIMScheduler

from aigi_detection.models.modules.soft_prompt import SoftPrompt, encode_plain, prompt_update_stats


def test_prompt_update_stopping_uses_actual_parameter_delta():
    before = torch.full((16, 768), 0.02)
    assert prompt_update_stats(before, before)['near_stationary']
    assert prompt_update_stats(before, before + 1e-6)['near_stationary']
    assert not prompt_update_stats(before, before + 1e-3)['near_stationary']
    assert prompt_update_stats(torch.zeros_like(before), torch.zeros_like(before))['relative_update_rms'] == 0


def test_bce_only_objective_has_one_generation_and_no_regularizers():
    objective = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/train_prompt.py'))['objective']
    parameter = nn.Parameter(torch.tensor(0.2))
    calls = []
    class SD:
        text_encoder = None
        def generate(self, conditioning, empty, seed, trace=None):
            calls.append(seed)
            return conditioning.sigmoid().expand(1, 3, 224, 224)
    class Critic(nn.Module):
        def forward(self, image):
            return image.mean((1, 2, 3), keepdim=False)[:, None]
    loss, logits = objective(SD(), lambda encoder: parameter, Critic(), None, 42)
    torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)))
    loss.backward()
    assert parameter.grad is not None and parameter.grad.abs() > 0
    assert calls == [42]


def test_perceptual_objective_uses_same_seed_frozen_reference_and_no_prompt_penalty():
    objective = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/train_prompt.py'))['objective']
    calls = []
    class Prompt(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.2))
        def forward(self, encoder):
            return self.value
        def regularization(self):
            raise AssertionError('Prompt penalty must not be evaluated')
    class SD:
        text_encoder = None
        def generate(self, conditioning, empty, seed, trace=None):
            calls.append((seed, torch.is_grad_enabled()))
            return conditioning.sigmoid().expand(1, 3, 224, 224)
    class Critic(nn.Module):
        def forward(self, image):
            return image.mean((1, 2, 3))[:, None]
    prompt = Prompt()
    components = {}
    phi = lambda image: (image.mean((2, 3)),)
    prefix = torch.tensor(-0.2, requires_grad=True)
    loss, logits = objective(SD(), prompt, Critic(), None, 42, perceptual=phi, prefix=prefix,
                             perceptual_weight=10., components=components)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    expected = bce + 10 * (prompt.value.sigmoid() - prefix.detach().sigmoid()).square()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prompt.value.grad is not None and prefix.grad is None
    assert calls == [(42, False), (42, True)]
    assert components['perceptual'] > 0


from aigi_detection.models.modules.differentiable_sd import DifferentiableSD


class Tokenizer:
    model_max_length = 24
    bos_token_id, eos_token_id, pad_token_id = 0, 63, 63

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = [1, 2, 3] if text else []
        if not add_special_tokens:
            return {'input_ids': ids}
        ids = [0] + ids + [63]
        mask = [1] * len(ids) + [0] * (self.model_max_length - len(ids))
        ids += [63] * (self.model_max_length - len(ids))
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([mask])}


def encoder():
    model = CLIPTextModel(CLIPTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=24,
                                       bos_token_id=0, eos_token_id=63, pad_token_id=63, attention_dropout=0.0))
    return model.eval().requires_grad_(False)


def test_only_soft_tokens_update_and_prefix_is_causal():
    model, tokenizer = encoder(), Tokenizer()
    prompt = SoftPrompt(model, tokenizer, tokens=16)
    assert list(dict(prompt.named_parameters())) == ['soft_tokens']
    before = {key: value.clone() for key, value in model.state_dict().items()}
    old_prompt = prompt.soft_tokens.detach().clone()
    plain = encode_plain(model, tokenizer, 'a realistic photo')
    condition = prompt(model)
    torch.testing.assert_close(condition[:, :prompt.start], plain[:, :prompt.start])
    condition.square().sum().backward()
    assert prompt.soft_tokens.grad is not None and prompt.soft_tokens.grad.norm() > 0
    optimizer = torch.optim.AdamW(prompt.parameters(), lr=1e-3, weight_decay=0)
    optimizer.step()
    assert not torch.equal(prompt.soft_tokens, old_prompt)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_prefix_only_and_reload(tmp_path):
    model, tokenizer = encoder(), Tokenizer()
    plain = encode_plain(model, tokenizer, 'a realistic photo')
    torch.testing.assert_close(SoftPrompt(model, tokenizer, tokens=0)(model), plain, rtol=0, atol=0)
    first = SoftPrompt(model, tokenizer)
    second = SoftPrompt(model, tokenizer, seed=99)
    torch.save(first.state_dict(), tmp_path / 'prompt.pt')
    second.load_state_dict(torch.load(tmp_path / 'prompt.pt', weights_only=True))
    torch.testing.assert_close(first(model), second(model), rtol=0, atol=0)
    assert len(model.get_input_embeddings()._forward_hooks) == 0


class TinyUNet(nn.Module):
    config = SimpleNamespace(in_channels=4)

    def forward(self, image, timestep, encoder_hidden_states):
        conditioning = encoder_hidden_states.mean((1, 2))[:, None, None, None]
        return SimpleNamespace(sample=image * 0.01 + conditioning * 0.1 + timestep * 0.0001)


class TinyVAE(nn.Module):
    config = SimpleNamespace(scaling_factor=1.0)

    def decode(self, image):
        return SimpleNamespace(sample=image[:, :3] * 0.1)


def test_checkpointed_full_sampler_matches_outputs_and_gradients():
    sd = DifferentiableSD.__new__(DifferentiableSD)
    sd.device, sd.dtype = torch.device('cpu'), torch.float32
    sd.steps, sd.guidance, sd.size = 4, 7.5, 32
    sd.unet, sd.vae = TinyUNet(), TinyVAE()
    sd.scheduler = DDIMScheduler(num_train_timesteps=20, clip_sample=False)
    condition = torch.randn(1, 4, 8, requires_grad=True)
    empty = torch.zeros_like(condition)
    first = sd.generate(condition, empty, 42, checkpointing=False)
    first_gradient = torch.autograd.grad(first.sum(), condition)[0]
    trace = {}
    second = sd.generate(condition, empty, 42, checkpointing=True, trace=trace)
    second_gradient = torch.autograd.grad(second.sum(), condition)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_gradient, second_gradient, rtol=0, atol=0)
    assert second_gradient.norm() > 0 and len(trace) == 3
