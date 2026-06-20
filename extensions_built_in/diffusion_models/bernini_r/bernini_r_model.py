import os
from typing import Any, Dict, List, Optional, Union

import torch
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision.transforms import functional as TF
from typing_extensions import Self

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.memory_management import MemoryManager
from toolkit.prompt_utils import PromptEmbeds
from toolkit.util.quantize import quantize_model
from toolkit.models.wan21.wan21 import Wan21

from ..wan22.wan22_14b_model import (
    Wan2214bModel,
    boundary_ratio_t2v,
)
from .src.models.transformer_wan import WanTransformer3DModel


class BerniniRDualTransformer3DModel(torch.nn.Module):
    def __init__(
        self,
        transformer_1: WanTransformer3DModel,
        transformer_2: WanTransformer3DModel,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        boundary_ratio: float = boundary_ratio_t2v,
        low_vram: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_1 = transformer_1
        self.transformer_2 = transformer_2
        self.torch_dtype = torch_dtype
        self.device_torch = device
        self.boundary_ratio = boundary_ratio
        self.boundary = self.boundary_ratio * 1000
        self.low_vram = low_vram
        self._active_transformer_name = "transformer_1"

    @property
    def device(self) -> torch.device:
        return self.device_torch

    @property
    def dtype(self) -> torch.dtype:
        return self.torch_dtype

    @property
    def config(self):
        return self.transformer.config

    @property
    def transformer(self) -> WanTransformer3DModel:
        return getattr(self, self._active_transformer_name)

    def enable_gradient_checkpointing(self):
        if hasattr(self.transformer_1, "enable_gradient_checkpointing"):
            try:
                self.transformer_1.enable_gradient_checkpointing()
            except ValueError:
                pass
        if hasattr(self.transformer_2, "enable_gradient_checkpointing"):
            try:
                self.transformer_2.enable_gradient_checkpointing()
            except ValueError:
                pass

    def get_transformer_for_timestep(self, timestep: torch.Tensor) -> tuple[str, WanTransformer3DModel]:
        with torch.no_grad():
            if timestep.float().mean().item() > self.boundary:
                t_name = "transformer_1"
            else:
                t_name = "transformer_2"

            if t_name != self._active_transformer_name:
                if self.low_vram:
                    getattr(self, self._active_transformer_name).to("cpu")
                    getattr(self, t_name).to(self.device_torch)
                    torch.cuda.empty_cache()
                self._active_transformer_name = t_name

        transformer = self.transformer
        if transformer.device != self.device_torch:
            if self.low_vram:
                other_tname = "transformer_1" if t_name == "transformer_2" else "transformer_2"
                getattr(self, other_tname).to("cpu")
            transformer.to(self.device_torch)
        return t_name, transformer

    def to(self, *args, **kwargs) -> Self:
        return self


class BerniniRModel(Wan2214bModel):
    arch = "bernini_r"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model_config=model_config,
            dtype=dtype,
            custom_pipeline=custom_pipeline,
            noise_scheduler=noise_scheduler,
            **kwargs,
        )
        self.target_lora_modules = ["BerniniRDualTransformer3DModel"]
        self.latent_space_version = "bernini_r"

    def load_wan_transformer(self, transformer_path, subfolder=None):
        if self.model_config.split_model_over_gpus:
            raise ValueError("Splitting model over gpus is not supported for Bernini-R")
        if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
            raise ValueError("Assistant LoRA is not supported for Bernini-R currently")
        if self.model_config.lora_path is not None:
            raise ValueError("Loading LoRA is not supported for Bernini-R currently")

        transformer_path_1 = transformer_path
        subfolder_1 = subfolder
        transformer_path_2 = transformer_path
        subfolder_2 = subfolder
        if subfolder_2 is None:
            transformer_path_2 = os.path.join(os.path.dirname(transformer_path_1), "transformer_2")
        else:
            subfolder_2 = "transformer_2"

        dtype = self.torch_dtype
        self.print_and_status_update("Loading Bernini-R transformer 1")
        transformer_1 = WanTransformer3DModel.from_pretrained(
            transformer_path_1,
            subfolder=subfolder_1,
            torch_dtype=dtype,
            use_src_id_rotary_emb=True,
        ).to(dtype=dtype)
        flush()

        if self.model_config.low_vram:
            transformer_1.to("cpu", dtype=dtype)
        else:
            transformer_1.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing Bernini-R transformer 1")
            quantize_model(self, transformer_1)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving Bernini-R transformer 1 to CPU")
            transformer_1.to("cpu")

        self.print_and_status_update("Loading Bernini-R transformer 2")
        transformer_2 = WanTransformer3DModel.from_pretrained(
            transformer_path_2,
            subfolder=subfolder_2,
            torch_dtype=dtype,
            use_src_id_rotary_emb=True,
        ).to(dtype=dtype)
        flush()

        if self.model_config.low_vram:
            transformer_2.to("cpu", dtype=dtype)
        else:
            transformer_2.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing Bernini-R transformer 2")
            quantize_model(self, transformer_2)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving Bernini-R transformer 2 to CPU")
            transformer_2.to("cpu")

        transformer = BerniniRDualTransformer3DModel(
            transformer_1=transformer_1,
            transformer_2=transformer_2,
            torch_dtype=self.torch_dtype,
            device=self.device_torch,
            boundary_ratio=boundary_ratio_t2v,
            low_vram=self.model_config.low_vram,
        )

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is not None:
            self.print_and_status_update("Applying Accuracy Recovery Adapter to Bernini-R transformers")
            quantize_model(self, transformer)
            flush()

        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            MemoryManager.attach(
                transformer_1,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_1.scale_shift_table] + [block.scale_shift_table for block in transformer_1.blocks],
            )
            MemoryManager.attach(
                transformer_2,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_2.scale_shift_table] + [block.scale_shift_table for block in transformer_2.blocks],
            )

        return transformer

    def load_model(self):
        Wan21.load_model(self)
        self.pipeline.transformer = None
        self.pipeline.transformer_2 = None

    def get_generation_pipeline(self):
        return self

    def set_progress_bar_config(self, *args, **kwargs):
        pass

    @staticmethod
    def _to_spatial(x: torch.Tensor, shape):
        return rearrange(
            x,
            "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
            t=shape[2],
            h=shape[3] // 2,
            w=shape[4] // 2,
            pt=1,
            ph=2,
            pw=2,
        )

    @staticmethod
    def _to_packed(x: torch.Tensor, shape):
        return rearrange(
            x,
            "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)",
            t=shape[2],
            h=shape[3] // 2,
            w=shape[4] // 2,
            pt=1,
            ph=2,
            pw=2,
        )

    def _get_source_latents(self, batch: DataLoaderBatchDTO) -> list[torch.Tensor]:
        if batch.control_tensor is None:
            return []

        controls = batch.control_tensor.to(self.device_torch, dtype=self.torch_dtype)
        if controls.ndim == 4:
            # B,C,H,W image controls.
            controls = controls * 2.0 - 1.0
            return list(self.encode_images([control for control in controls]).unbind(0))
        if controls.ndim == 5:
            # B,T,C,H,W video controls from the dataloader.
            controls = controls * 2.0 - 1.0
            return list(self.encode_images([control for control in controls]).unbind(0))
        raise ValueError(f"Unsupported Bernini-R control tensor shape {controls.shape}")

    @torch.no_grad()
    def _encode_control_path(self, control_path: str, width: int, height: int, num_frames: int):
        ext = os.path.splitext(control_path)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv"]:
            import cv2

            cap = cv2.VideoCapture(control_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open control video {control_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frame_index = total_frames - 1
            interval = max_frame_index / (num_frames - 1) if num_frames > 1 else 0
            frame_indices = [min(int(round(i * interval)), max_frame_index) for i in range(num_frames)]
            frames = []
            try:
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError(f"Could not read frame {frame_idx} from {control_path}")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame).convert("RGB").resize((width, height), Image.LANCZOS)
                    frames.append(TF.to_tensor(img))
            finally:
                cap.release()
            tensor = torch.stack(frames)
        else:
            img = Image.open(control_path).convert("RGB").resize((width, height), Image.LANCZOS)
            tensor = TF.to_tensor(img)

        tensor = tensor.to(self.device_torch, dtype=self.torch_dtype) * 2.0 - 1.0
        return self.encode_images([tensor])[0:1]

    @torch.no_grad()
    def _sample_latents(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        source_latent: Optional[torch.Tensor] = None,
    ):
        scheduler = self.get_train_scheduler()
        scheduler.set_timesteps(num_inference_steps, device=self.device_torch)
        timesteps = scheduler.timesteps.to(self.device_torch)

        num_frames = num_frames // 4 * 4 + 1
        num_channels_latents = self.model.transformer_1.config.in_channels
        num_latent_frames = (num_frames - 1) // 4 + 1
        shape = (
            1,
            num_channels_latents,
            num_latent_frames,
            int(height) // 8,
            int(width) // 8,
        )
        latents = randn_tensor(shape, device=self.device_torch, dtype=torch.float32, generator=generator)

        for t in timesteps:
            timestep = t.expand(1)
            _, transformer = self.model.get_transformer_for_timestep(timestep)
            cond_pred = self._predict_single_sample(
                transformer=transformer,
                target_latent=latents.to(self.torch_dtype),
                timestep=timestep,
                text_embed=prompt_embeds,
                source_latent=source_latent,
            )
            uncond_pred = self._predict_single_sample(
                transformer=transformer,
                target_latent=latents.to(self.torch_dtype),
                timestep=timestep,
                text_embed=negative_prompt_embeds,
                source_latent=source_latent,
            )
            noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents

    def _predict_single_sample(
        self,
        transformer: WanTransformer3DModel,
        target_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embed: torch.Tensor,
        source_latent: Optional[torch.Tensor] = None,
    ):
        cond_latents = []
        cond_rotary = []
        cond_masks = []
        cond_len = 0
        if source_latent is not None:
            src_tokens, src_rotary = transformer.patch_vae_latent(
                source_latent.to(self.device_torch, dtype=self.torch_dtype),
                source_id=1.0,
            )
            cond_latents.append(src_tokens)
            cond_rotary.append(src_rotary)
            cond_masks.append(torch.zeros(src_tokens.shape[1], device=self.device_torch, dtype=torch.bool))
            cond_len += src_tokens.shape[1]

        target_tokens, target_rotary = transformer.patch_vae_latent(target_latent, source_id=0)
        target_mask = torch.ones(target_tokens.shape[1], device=self.device_torch, dtype=torch.bool)
        latent_input = torch.cat(cond_latents + [target_tokens], dim=1).to(self.torch_dtype)
        rotary_emb = torch.cat(cond_rotary + [target_rotary], dim=2)
        vae_mask = torch.cat(cond_masks + [target_mask], dim=0)
        total_len = cond_len + target_tokens.shape[1]

        pred = transformer(
            hidden_states=latent_input,
            timestep=timestep,
            encoder_hidden_states=text_embed,
            rotary_emb=rotary_emb,
            batch_image_vae_seqlen=[total_len],
            text_features_length=[text_embed.shape[1]],
            return_dict=False,
        )[0][:, vae_mask, :]
        return self._to_spatial(pred, target_latent.shape)

    @torch.no_grad()
    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        num_frames = (gen_config.num_frames - 1) // 4 * 4 + 1
        d = self.get_bucket_divisibility()
        height = gen_config.height // d * d
        width = gen_config.width // d * d

        source_latent = None
        if gen_config.ctrl_img is not None:
            source_latent = self._encode_control_path(gen_config.ctrl_img, width, height, num_frames)

        latents = self._sample_latents(
            prompt_embeds=conditional_embeds.text_embeds.to(self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(self.device_torch, dtype=self.torch_dtype),
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            generator=generator,
            source_latent=source_latent,
        )
        output = self.decode_latents(latents)[0].permute(1, 0, 2, 3)
        images = []
        for frame in output:
            frame = ((frame.float().clamp(-1, 1) + 1) / 2).cpu()
            images.append(TF.to_pil_image(frame))
        return images if num_frames > 1 else images[0]

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs,
    ):
        if batch is None:
            raise ValueError("Bernini-R training requires batch metadata")

        timestep = timestep.to(self.device_torch)
        latent_model_input = latent_model_input.to(self.device_torch, dtype=self.torch_dtype)
        text_embeds = text_embeddings.text_embeds.to(self.device_torch, dtype=self.torch_dtype)

        _, transformer = self.model.get_transformer_for_timestep(timestep)
        source_latents = self._get_source_latents(batch)
        batch_size = latent_model_input.shape[0]
        if len(source_latents) not in (0, batch_size):
            if batch_size == len(source_latents) * 2:
                source_latents = source_latents + source_latents
            else:
                raise ValueError(
                    f"Expected 0 or {batch_size} Bernini-R control latents, got {len(source_latents)}"
                )

        outputs = []
        for idx in range(batch_size):
            target_latent = latent_model_input[idx : idx + 1]
            text_embed = text_embeds[idx : idx + 1]
            t = timestep[idx : idx + 1] if timestep.ndim == 1 else timestep[idx : idx + 1]

            source_latent = source_latents[idx].unsqueeze(0) if source_latents else None
            outputs.append(self._predict_single_sample(transformer, target_latent, t, text_embed, source_latent))

        return torch.cat(outputs, dim=0)

    def save_model(self, output_path, meta, save_dtype):
        Wan2214bModel.save_model(self, output_path, meta, save_dtype)

    def save_lora(
        self,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.network.network_config.split_multistage_loras:
            save_file(state_dict, output_path, metadata=metadata)
            return

        high_noise_lora = {}
        low_noise_lora = {}
        only_train_high_noise = self.train_high_noise and not self.train_low_noise
        only_train_low_noise = self.train_low_noise and not self.train_high_noise

        for key in state_dict:
            if ".transformer_1." in key or only_train_high_noise:
                high_noise_lora[key.replace(".transformer_1.", ".")] = state_dict[key]
            elif ".transformer_2." in key or only_train_low_noise:
                low_noise_lora[key.replace(".transformer_2.", ".")] = state_dict[key]

        if len(high_noise_lora.keys()) > 0:
            save_file(high_noise_lora, output_path.replace(".safetensors", "_high_noise.safetensors"), metadata=metadata)
        if len(low_noise_lora.keys()) > 0:
            save_file(low_noise_lora, output_path.replace(".safetensors", "_low_noise.safetensors"), metadata=metadata)

    def load_lora(self, file: str):
        if "_high_noise.safetensors" not in file and "_low_noise.safetensors" not in file:
            return load_file(file)

        high_noise_lora_path = file.replace("_low_noise.safetensors", "_high_noise.safetensors")
        low_noise_lora_path = file.replace("_high_noise.safetensors", "_low_noise.safetensors")
        combined_dict = {}

        if os.path.exists(high_noise_lora_path) and self.train_high_noise:
            for key, value in load_file(high_noise_lora_path).items():
                combined_dict[key.replace("diffusion_model.", "diffusion_model.transformer_1.")] = value
        if os.path.exists(low_noise_lora_path) and self.train_low_noise:
            for key, value in load_file(low_noise_lora_path).items():
                combined_dict[key.replace("diffusion_model.", "diffusion_model.transformer_2.")] = value

        if not self.train_high_noise or not self.train_low_noise:
            new_dict = {}
            for key in combined_dict:
                new_key = key.replace(".transformer_1.", ".").replace(".transformer_2.", ".")
                new_dict[new_key] = combined_dict[key]
            combined_dict = new_dict

        return combined_dict
