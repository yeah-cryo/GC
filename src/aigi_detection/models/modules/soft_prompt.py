"""Soft token insertion through CLIP's ordinary embedding/attention forward."""
import torch
from torch import nn


def prompt_update_stats(before, after, absolute_tolerance=1e-6, relative_tolerance=1e-3):
    """Measure the actual optimizer update, rather than noisy loss convergence."""
    delta = (after.detach() - before.detach()).float().square().mean().sqrt().item()
    scale = before.detach().float().square().mean().sqrt().item()
    threshold = absolute_tolerance + relative_tolerance * scale
    return {'update_rms': delta, 'relative_update_rms': delta / max(scale, 1e-12),
            'update_threshold': threshold, 'near_stationary': delta <= threshold}


def encode_plain(encoder, tokenizer, text):
    encoded = tokenizer(text, padding='max_length', max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors='pt')
    device = next(encoder.parameters()).device
    kwargs = {'input_ids': encoded['input_ids'].to(device)}
    if getattr(encoder.config, 'use_attention_mask', False):
        kwargs['attention_mask'] = encoded['attention_mask'].to(device)
    return encoder(**kwargs).last_hidden_state


class SoftPrompt(nn.Module):
    def __init__(self, encoder, tokenizer, prefix='a realistic photo', tokens=16, seed=42, noise_std=0.001):
        super().__init__()
        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        length = tokenizer.model_max_length
        if len(prefix_ids) + tokens + 2 > length:
            raise ValueError('Prefix and soft tokens exceed the CLIP context length')
        self.start = 1 + len(prefix_ids)
        self.tokens = tokens
        rng = torch.Generator().manual_seed(seed)
        embedding = encoder.get_input_embeddings().weight
        excluded = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
        vocabulary = torch.tensor([i for i in range(embedding.shape[0]) if i not in excluded])
        sampled = vocabulary[torch.randint(len(vocabulary), (tokens,), generator=rng)]
        initial = embedding.detach()[sampled.to(embedding.device)].float().cpu().clone()
        initial += torch.randn(initial.shape, generator=rng) * noise_std
        self.soft_tokens = nn.Parameter(initial.clone())
        self.register_buffer('initial_tokens', initial)
        ids = [tokenizer.bos_token_id] + prefix_ids + sampled.tolist() + [tokenizer.eos_token_id]
        mask = [1] * len(ids) + [0] * (length - len(ids))
        ids += [tokenizer.pad_token_id] * (length - len(ids))
        self.register_buffer('input_ids', torch.tensor([ids], dtype=torch.long))
        self.register_buffer('attention_mask', torch.tensor([mask], dtype=torch.long))

    def forward(self, encoder, initialized=False):
        values = self.initial_tokens if initialized else self.soft_tokens
        def replace_embedding(module, inputs, output):
            soft = values.to(output.dtype).unsqueeze(0).expand(output.shape[0], -1, -1)
            return torch.cat((output[:, :self.start], soft, output[:, self.start + self.tokens:]), dim=1)
        hook = encoder.get_input_embeddings().register_forward_hook(replace_embedding)
        try:
            kwargs = {'input_ids': self.input_ids}
            if getattr(encoder.config, 'use_attention_mask', False):
                kwargs['attention_mask'] = self.attention_mask
            return encoder(**kwargs).last_hidden_state
        finally:
            hook.remove()

    def regularization(self):
        return (self.soft_tokens - self.initial_tokens).square().mean()
