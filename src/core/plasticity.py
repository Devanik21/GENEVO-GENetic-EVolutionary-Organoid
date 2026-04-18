"""
Local plasticity rules and implementation.
"""
import torch

def apply_hebbian_update(module, rule):
    """Apply Hebbian learning rule."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'weight' in name and param.grad is not None:
                if hasattr(module, 'recent_activity') and module.recent_activity is not None:
                    activity_scale = module.recent_activity.abs().mean()
                    hebbian_delta = rule.learning_rate * activity_scale * torch.randn_like(param) * 0.01
                    param.add_(hebbian_delta)
                    if rule.decay > 0:
                        param.mul_(1 - rule.decay)
