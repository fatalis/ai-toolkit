# AI Toolkit Krea 2 Raw Sampling Fork

This fork is based on Ostris AI Toolkit and focuses on Krea 2 raw sampling.

The raw Krea 2 UI preset now adds sample-time LoRAs by default:

- Krea 2 Turbo LoRA
- Krea 2 TextFusion / refusal reduction LoRA

When you select the `Krea 2 (raw)` model preset in the UI, the generated job config includes:

```yaml
sample:
  loras:
    - path: "/path/to/krea2_turbo_lora_rank_64_bf16.safetensors"
      strength: 0.6
    - path: "/path/to/krea2-Krea2_TextFusion_Refusal_Reduction.safetensors"
      strength: 1
  guidance_scale: 1
  sample_steps: 8
```

These LoRAs are only applied during sampling. They do not replace the training network and they do not apply to the Krea 2 Turbo or edit-training presets.

## Changing LoRA Paths

The default paths are placeholders. Change them in your job config before running training or sampling.

In the UI, select the `Krea 2 (raw)` preset, then open the advanced config editor and find:

```yaml
sample:
  loras:
    - path: "/path/to/krea2_turbo_lora_rank_64_bf16.safetensors"
      strength: 0.6
    - path: "/path/to/krea2-Krea2_TextFusion_Refusal_Reduction.safetensors"
      strength: 1
```

Replace each `path` with the local path to your `.safetensors` file.

Example:

```yaml
sample:
  loras:
    - path: "/home/user/models/loras/krea2_turbo_lora_rank_64_bf16.safetensors"
      strength: 0.6
    - path: "/home/user/models/loras/krea2-Krea2_TextFusion_Refusal_Reduction.safetensors"
      strength: 1
```

You can also adjust `strength` values in the same config.

## Config Format

This fork uses only the `sample.loras` array for sample LoRAs.

Supported forms:

```yaml
sample:
  loras:
    - "/path/to/lora.safetensors"
```

or:

```yaml
sample:
  loras:
    - path: "/path/to/lora.safetensors"
      strength: 0.6
```

The older `sample.lora_path` and `sample.lora_strength` fields are not supported in this fork.
