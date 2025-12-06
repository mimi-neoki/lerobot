# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("vla0")
@dataclass
class VLA0Config(PreTrainedConfig):
    """Configuration for the VLA0 policy wrapper.

    The defaults mirror the reference vla0 configuration while keeping the API
    aligned with other LeRobot policies so it can be trained with `lerobot-train`.
    """

    # Input / output structure
    n_obs_steps: int = 1  # history length used for vision
    horizon: int = 8

    # Vision settings
    num_cameras: int = 1
    # (height, width) expected by Qwen processor; will be overridden by dataset shape when
    # `auto_set_rgb_from_dataset` is True.
    rgb_img_size: tuple[int, int] = (224, 224)
    auto_set_rgb_from_dataset: bool = True
    tiled_rgb_imgs: bool = True
    add_vision_id: bool = True

    # Qwen / LoRA options
    qwen_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    use_lora: bool = True
    use_qlora: bool = False
    lora_config: str = "default"
    lora_rank: int = 8
    use_flash_attention_2: bool = False
    use_amp: bool = True

    # Action tokenisation
    num_bins_actions: int = 1000
    action_mask_aug_per: float = 0.4

    # Optim presets
    optimizer_lr: float = 5e-6
    optimizer_weight_decay: float = 1e-10
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 0.0

    # Normalisation map (kept for consistency with other policies)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        # Ensure horizon and n_obs_steps are positive
        if self.n_obs_steps < 1:
            raise ValueError("`n_obs_steps` must be >= 1 for VLA0.")
        if self.horizon < 1:
            raise ValueError("`horizon` must be >= 1 for VLA0.")

    # ---- Presets ----
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        # Reference implementation uses a constant LR by default.
        return None

    # ---- Validation ----
    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError("VLA0 requires at least one visual observation.")
        # Ensure all image shapes match for stacking
        first_key, first_ft = next(iter(self.image_features.items()))
        for key, ft in self.image_features.items():
            if ft.shape != first_ft.shape:
                raise ValueError(f"Image feature {key} shape {ft.shape} != {first_key} shape {first_ft.shape}")

    # ---- Delta indices used by dataloader ----
    @property
    def observation_delta_indices(self) -> list:
        # e.g. n_obs_steps=1 -> [0]; n_obs_steps=2 -> [-1, 0]
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
