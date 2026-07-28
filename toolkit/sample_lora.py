import os
from typing import TYPE_CHECKING, Dict, Tuple

import torch
from safetensors.torch import load_file

from toolkit.kohya_lora import LoRANetwork
from toolkit.lora_special import LoRASpecialNetwork

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion


def _get_module_dimensions(state_dict: dict) -> Tuple[Dict[str, int], Dict[str, object]]:
    """Extract per-module ranks and alphas from native and PEFT LoRA weights."""
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, object] = {}

    for key, value in state_dict.items():
        for suffix in ('.lora_down.weight', '.lora_A.weight'):
            if key.endswith(suffix):
                module_name = key[:-len(suffix)]
                modules_dim[module_name] = int(value.shape[0])
                modules_dim[module_name.replace('.', '$$')] = int(value.shape[0])
                break

    for module_name, dim in modules_dim.items():
        alpha = state_dict.get(f'{module_name}.alpha', dim)
        if isinstance(alpha, torch.Tensor) and alpha.numel() == 1:
            alpha = alpha.detach().float().item()
        modules_alpha[module_name] = alpha

    if not modules_dim:
        raise ValueError('No LoRA weights were found in the sample LoRA file')

    return modules_dim, modules_alpha


def _filter_lora_state_dict(state_dict: dict) -> dict:
    supported_suffixes = (
        '.lora_down.weight',
        '.lora_up.weight',
        '.lora_A.weight',
        '.lora_B.weight',
        '.alpha',
    )
    return {
        key: value
        for key, value in state_dict.items()
        if key.endswith(supported_suffixes)
    }


def load_sample_lora_from_path(
        lora_path: str,
        sd: 'StableDiffusion',
) -> LoRASpecialNetwork:
    """Attach a local LoRA to a loaded model, initially inactive and on CPU."""
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f'Sample LoRA file not found: {lora_path}')
    if not lora_path.lower().endswith('.safetensors'):
        raise ValueError('Sample LoRA must be a .safetensors file')

    raw_state_dict = load_file(lora_path)
    state_dict = sd.convert_lora_weights_before_load(raw_state_dict)
    modules_dim, modules_alpha = _get_module_dimensions(state_dict)

    has_text_encoder_weights = any(
        name.startswith(('lora_te', 'text_encoder')) for name in modules_dim
    )
    target_lin_modules = list(
        getattr(sd, 'target_lora_modules', None)
        or LoRANetwork.UNET_TARGET_REPLACE_MODULE
    )
    transformer_only = not has_text_encoder_weights
    if (
            transformer_only
            and getattr(sd, 'get_base_model_version', lambda: None)() == 'krea2'
            and any('$$txtfusion$$' in name or '.txtfusion.' in name for name in modules_dim)
    ):
        # modules_dim already limits creation to exact checkpoint keys. Krea2's
        # txtfusion LoRAs live outside the main "blocks" path, so do not apply
        # the generic transformer block-name filter when loading them for samples.
        transformer_only = False

    network = LoRASpecialNetwork(
        text_encoder=sd.text_encoder,
        unet=sd.get_model_to_train(),
        multiplier=1.0,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        train_unet=True,
        train_text_encoder=has_text_encoder_weights,
        use_text_encoder_1=sd.model_config.use_text_encoder_1,
        use_text_encoder_2=sd.model_config.use_text_encoder_2,
        is_sdxl=sd.model_config.is_xl or sd.model_config.is_ssd,
        is_v2=sd.model_config.is_v2,
        is_v3=sd.model_config.is_v3,
        is_pixart=sd.model_config.is_pixart,
        is_auraflow=sd.model_config.is_auraflow,
        is_flux=sd.model_config.is_flux,
        is_lumina2=sd.model_config.is_lumina2,
        is_ssd=sd.model_config.is_ssd,
        is_vega=sd.model_config.is_vega,
        is_transformer=sd.is_transformer,
        transformer_only=transformer_only,
        target_lin_modules=target_lin_modules,
        base_model=sd,
    )
    network.apply_to(
        sd.text_encoder,
        sd.get_model_to_train(),
        apply_text_encoder=has_text_encoder_weights,
        apply_unet=True,
    )
    # load_weights owns the actual model-specific conversion and key mapping;
    # pass the original keys here rather than the inspection copy above.
    network.load_weights(_filter_lora_state_dict(raw_state_dict))
    network.eval()
    network.is_active = False
    network.force_to('cpu', dtype=sd.torch_dtype)
    network._update_torch_multiplier()
    return network
