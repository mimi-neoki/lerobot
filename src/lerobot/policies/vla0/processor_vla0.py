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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.policies.vla0.configuration_vla0 import VLA0Config
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    create_transition,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_IMAGES, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@ProcessorStepRegistry.register("vla0_pack_inputs")
@dataclass
class VLA0PackInputsProcessor(ProcessorStep):
    """Stack camera images and expose them as an RGB tensor expected by vla0."""

    history: int
    num_cameras: int | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))

        image_keys = sorted(
            [k for k in observation if k.startswith(OBS_IMAGES) and not k.endswith("_is_pad")]
        )
        if len(image_keys) == 0:
            raise ValueError("VLA0 requires at least one camera observation (keys starting with observation.images).")
        if self.num_cameras is not None and len(image_keys) != self.num_cameras:
            # Keep going but surface a clear message for misconfigured datasets.
            raise ValueError(f"Expected {self.num_cameras} cameras, found {len(image_keys)}: {image_keys}")

        stacked_cams = []
        for key in image_keys:
            tensor = observation[key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)

            # Ensure shape (B, history, C, H, W)
            if tensor.ndim == 4:  # (B, C, H, W)
                tensor = tensor.unsqueeze(1)
            if tensor.ndim != 5:
                raise ValueError(f"Image tensor for {key} should have 5 dims after batching, got {tensor.shape}.")

            # Align history length with config (pad with last frame or trim)
            if tensor.shape[1] < self.history:
                pad = tensor[:, -1:].repeat(1, self.history - tensor.shape[1], 1, 1, 1)
                tensor = torch.cat([tensor, pad], dim=1)
            elif tensor.shape[1] > self.history:
                tensor = tensor[:, -self.history :, ...]

            # Move channels last and convert to uint8 in [0, 255]
            tensor = tensor.permute(0, 1, 3, 4, 2).contiguous()
            if tensor.is_floating_point():
                if tensor.max() <= 1.01:
                    tensor = tensor * 255.0
                tensor = tensor.clamp(0, 255).round()
            tensor = tensor.to(torch.uint8)
            stacked_cams.append(tensor)

        rgb = torch.stack(stacked_cams, dim=2)  # (B, history, num_cam, H, W, 3)
        complementary["rgb"] = rgb

        return create_transition(
            observation=observation,
            action=transition.get(TransitionKey.ACTION),
            reward=transition.get(TransitionKey.REWARD, 0.0),
            done=transition.get(TransitionKey.DONE, False),
            truncated=transition.get(TransitionKey.TRUNCATED, False),
            info=transition.get(TransitionKey.INFO, {}),
            complementary_data=complementary,
        )

    def transform_features(self, features: dict[TransitionKey, Any]) -> dict[TransitionKey, Any]:
        return features


def make_vla0_pre_post_processors(
    config: VLA0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create processors for VLA0 to keep parity with other policies."""

    num_cameras = len(config.image_features) if config.image_features else config.num_cameras
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # Keep identical to other policies for save/load parity
        VLA0PackInputsProcessor(history=config.n_obs_steps, num_cameras=num_cameras),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
