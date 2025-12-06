#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team.
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

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla0.configuration_vla0 import VLA0Config
from lerobot.utils.constants import ACTION


def _ensure_vla0_on_path():
    """Add the vendored vla0 package to sys.path if it is not importable."""
    try:
        import rv_train  # noqa: F401
        return
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        vla0_root = repo_root / "vla0"
        if vla0_root.exists() and str(vla0_root) not in sys.path:
            sys.path.append(str(vla0_root))


class VLA0Policy(PreTrainedPolicy):
    """Thin wrapper that exposes the reference vla0 Qwen actor through the LeRobot API."""

    config_class = VLA0Config
    name = "vla0"

    def __init__(self, config: VLA0Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        _ensure_vla0_on_path()
        try:
            from rv_train.models.qwen.model import QwenActor
            import rv_train.constants as C
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise ImportError(
                "rv_train (vla0) modules are missing. Install extras with `pip install -e '.[vla0]'`."
            ) from exc

        action_dim = self.config.action_feature.shape[0] if self.config.action_feature else None
        if action_dim is None:
            raise ValueError("VLA0Config.output_features must include the action feature.")

        num_cameras = len(self.config.image_features) if self.config.image_features else self.config.num_cameras
        self.config.num_cameras = num_cameras

        # If requested, infer image size from dataset feature shapes.
        if self.config.auto_set_rgb_from_dataset and self.config.image_features:
            first_image = next(iter(self.config.image_features.values()))
            # PolicyFeature.shape is (C, H, W)
            if len(first_image.shape) == 3:
                _, h, w = first_image.shape
                self.config.rgb_img_size = (h, w)

        self.actor = QwenActor(
            qwen_model_id=self.config.qwen_model_id,
            action_type=C.ORIGINAL,
            original_action_dim=action_dim,
            horizon=self.config.horizon,
            history=self.config.n_obs_steps,
            use_lora=self.config.use_lora,
            use_qlora=self.config.use_qlora,
            num_cam=num_cameras,
            lora_config=self.config.lora_config,
            lora_rank=self.config.lora_rank,
            rgb_input=True,
            rgb_img_size=self.config.rgb_img_size,
            add_vision_id=self.config.add_vision_id,
            tiled_rgb_imgs=self.config.tiled_rgb_imgs,
            num_bins_actions=self.config.num_bins_actions,
            use_flash_attention_2=self.config.use_flash_attention_2,
            action_mask_aug_per=self.config.action_mask_aug_per,
        )
        self.actor.to(self.config.device)
        # Provide a sane default to satisfy the actor; real stats will override this when available.
        self.actor.set_dataset_stats({"out_ori_act": {"min": [-1.0] * action_dim, "max": [1.0] * action_dim}})

    # ---- Training / optimisation helpers ----
    def get_optim_params(self) -> dict[str, Any]:
        # Return a flat list of trainable parameters.
        return [p for p in self.parameters() if p.requires_grad]

    def set_dataset_stats(self, dataset_stats: dict | None):
        """Adapt LeRobot stats to vla0's expected format."""
        if not dataset_stats:
            return
        stats = {}
        if ACTION in dataset_stats:
            stats["out_ori_act"] = dataset_stats[ACTION]
        elif "out_ori_act" in dataset_stats:
            stats = dataset_stats
        if stats:
            self.actor.set_dataset_stats(stats)

    # ---- Core forward ----
    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict | None]:
        rgb = batch.get("rgb")
        if rgb is None:
            raise ValueError("VLA0Policy expects `rgb` in the batch. Check the preprocessor.")

        instr = batch.get("task", None)
        if instr is None:
            instr = [""] * rgb.shape[0]
        if isinstance(instr, str):
            instr = [instr]

        target_actions = batch.get(ACTION)
        if target_actions is None:
            raise ValueError("Training VLA0 requires ground-truth actions in the batch.")
        if target_actions.dim() == 2:
            target_actions = target_actions.unsqueeze(1)
        if target_actions.shape[1] < self.config.horizon:
            pad = target_actions[:, -1:].repeat(1, self.config.horizon - target_actions.shape[1], 1)
            target_actions = torch.cat([target_actions, pad], dim=1)
        elif target_actions.shape[1] > self.config.horizon:
            target_actions = target_actions[:, : self.config.horizon]

        outputs = self.actor(
            instr=list(instr),
            rgb=rgb,
            out_ori_act=target_actions,
            get_loss=True,
            get_action=False,
            pc=None,
            rgb_pc=None,
            ori_act=None,
            ee_pos=None,
            ee_rot=None,
            ee_gri=None,
            out_ee_pos=None,
            out_ee_rot=None,
            out_ee_gri=None,
        )
        loss = outputs["loss"]
        return loss, {"loss": loss.item()}

    # ---- Inference helpers ----
    def _generate(self, batch: dict[str, Any], one_step: bool = False, last_action_txt: str = "") -> torch.Tensor:
        rgb = batch.get("rgb")
        instr = batch.get("task", [""] * rgb.shape[0])
        if isinstance(instr, str):
            instr = [instr]
        outputs = self.actor(
            instr=list(instr),
            rgb=rgb,
            get_loss=False,
            get_action=True,
            get_one_step_action=one_step,
            last_action_txt=last_action_txt,
            pc=None,
            rgb_pc=None,
            ori_act=None,
            ee_pos=None,
            ee_rot=None,
            ee_gri=None,
            out_ori_act=None,
            out_ee_pos=None,
            out_ee_rot=None,
            out_ee_gri=None,
        )
        return outputs["out_ori_act"]

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        return self._generate(batch, one_step=False)

    def select_action(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        # Return only the first step for env stepping; keep the whole chunk if downstream wants it.
        chunk = self._generate(batch, one_step=False)
        return chunk[:, 0]

    def reset(self):
        # No recurrent state to clear for vla0.
        return
