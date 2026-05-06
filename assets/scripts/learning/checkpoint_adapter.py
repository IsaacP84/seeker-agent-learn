from typing import List, Tuple

import torch
from itertools import zip_longest


def _normalize_keys(sd: dict) -> dict:
    """Remove leading 'module.' from keys if present."""
    if any(k.startswith('module.') for k in sd.keys()):
        return { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
    return dict(sd)


def _adapt_tensor_to_shape(tensor: torch.Tensor, target_shape: tuple, dtype=None, device=None) -> torch.Tensor:
    """Return a tensor reshaped by truncation or zero-padding to match target_shape.

    Copies the overlapping slice from `tensor` into a new tensor with `target_shape`.
    """
    if dtype is None:
        dtype = tensor.dtype
    if device is None:
        device = tensor.device

    target = torch.zeros(target_shape, dtype=dtype, device=device)

    # compute overlapping region
    overlap = tuple(min(s_old, s_new) for s_old, s_new in zip(tensor.shape, target_shape))
    if not overlap:
        return target

    src_slices = tuple(slice(0, o) for o in overlap)
    dst_slices = src_slices

    # ensure tensor is on same dtype/device for copying
    t = tensor.to(dtype=dtype, device=device)
    target[dst_slices] = t[src_slices]
    return target


def _build_state_layout(config: dict) -> List[Tuple[str, int]]:
    """Return the ordered state input layout segments for a given env config."""
    raycast = config.get('raycast', {})
    sighting = config.get('sighting', {})
    sizes = config.get('sizes', {})

    num_rays = int(raycast.get('num_rays', 0))
    num_ground_rays = int(raycast.get('num_ground_rays', 0))
    sighting_history = int(sighting.get('history', 0))
    num_actions = int(sizes.get('num_actions', 0))
    action_history = int(sizes.get('action_history', 0))

    return [
        ('ray_hits', num_rays),
        ('ray_interest', num_rays),
        ('sightings', num_rays * sighting_history * 4),
        ('ground_rays', num_ground_rays),
        ('goal_dir', 2),
        ('velocity', 4),
        ('angle', 2),
        ('goal_dist', 1),
        ('last_known_goal', 2),
        ('goal_visible', 1),
        ('staleness', 1),
        ('in_air_jumping', 2),
        ('prev_actions', num_actions * action_history),
    ]


def _adapt_input_weight_by_layout(weight: torch.Tensor, old_config: dict, new_config: dict, dtype=None, device=None) -> torch.Tensor:
    """Adapt the first linear input weight using state layout segment alignment."""
    if dtype is None:
        dtype = weight.dtype
    if device is None:
        device = weight.device

    old_layout = _build_state_layout(old_config)
    new_layout = _build_state_layout(new_config)

    old_size = sum(size for _, size in old_layout)
    new_size = sum(size for _, size in new_layout)
    if weight.shape[1] != old_size:
        raise ValueError(f'Expected first-layer input size {old_size} from old config, got {weight.shape[1]}')

    if [name for name, _ in old_layout] != [name for name, _ in new_layout]:
        # Fall back to simple padding/truncation if layout names diverge.
        return _adapt_tensor_to_shape(weight, (weight.shape[0], new_size), dtype=dtype, device=device)

    adapted = torch.zeros((weight.shape[0], new_size), dtype=dtype, device=device)
    old_index = 0
    new_index = 0

    for (_, old_segment_size), (_, new_segment_size) in zip_longest(old_layout, new_layout, fillvalue=(None, 0)):
        overlap = min(old_segment_size, new_segment_size)
        if overlap > 0:
            adapted[:, new_index:new_index+overlap] = weight[:, old_index:old_index+overlap].to(dtype=dtype, device=device)
        old_index += old_segment_size
        new_index += new_segment_size

    return adapted


def load_state_dict_with_adapt(model: torch.nn.Module, ckpt_state_dict: dict, old_config: dict = None, new_config: dict = None):
    """Load a checkpoint state_dict into `model`, adapting parameter tensors when shapes differ.

    If `old_config` and `new_config` are provided, the first input linear layer is
    adapted using state layout segments so new input blocks are inserted in the
    correct place (for example, expanded action history inputs).
    """
    if not isinstance(ckpt_state_dict, dict):
        raise TypeError('ckpt_state_dict must be a dict of tensors')

    ckpt = _normalize_keys(ckpt_state_dict)
    model_state = model.state_dict()
    adapted = {}

    first_weight_key = next((k for k in model_state if k.endswith('.weight')), None)

    for k, m_val in model_state.items():
        if k not in ckpt:
            continue
        v = ckpt[k]
        if v.shape == m_val.shape:
            adapted[k] = v.to(dtype=m_val.dtype, device=m_val.device)
            continue

        if old_config is not None and new_config is not None and k == first_weight_key and v.ndim == 2 and m_val.ndim == 2:
            try:
                adapted[k] = _adapt_input_weight_by_layout(v, old_config, new_config, dtype=m_val.dtype, device=m_val.device)
                continue
            except Exception:
                pass

        try:
            adapted[k] = _adapt_tensor_to_shape(v, tuple(m_val.shape), dtype=m_val.dtype, device=m_val.device)
        except Exception:
            continue

    model.load_state_dict(adapted, strict=False)
