# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Convert RobustSAM checkpoints from the original repository.

URL: https://github.com/robustsam/RobustSAM/tree/main.

"""

import argparse
import re

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    RobustSamConfig,
    RobustSamImageProcessor,
    RobustSamModel,
    RobustSamProcessor,
    RobustSamVisionConfig,
    RobustSamMaskDecoderConfig,
)

# Copied from transformers.models.sam.convert_sam_to_hf.get_config with Sam->RobustSam, and load the RobustSAM ckpts
def get_config(model_name):
    if "robustsam_vit_b" in model_name:
        vision_config = RobustSamVisionConfig()
        decoder_config = RobustSamMaskDecoderConfig(vit_dim=768)
    elif "robustsam_vit_l" in model_name:
        vision_config = RobustSamVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )
        decoder_config = RobustSamMaskDecoderConfig(vit_dim=1024)
    elif "robustsam_vit_h" in model_name:
        vision_config = RobustSamVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )
        decoder_config = RobustSamMaskDecoderConfig(vit_dim=1280)


    config = RobustSamConfig(
        vision_config=vision_config,
        mask_decoder_config = decoder_config,
    )

    return config

# Copied from transformers.models.sam.convert_sam_to_hf.KEYS_TO_MODIFY_MAPPING
KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "image_encoder": "vision_encoder",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "patch_embed.proj": "patch_embed.projection",
    ".norm": ".layer_norm",
    "blocks": "layers",
    "excitation.0": "excitation_fc1",
    "excitation.2": "excitation_fc2",
    "dnc_block_combined.SEMBlock.maxpool_conv.0.": "dnc_block_combined.SEMBlock.maxpool_conv.",
    "fourier_last_layer_features.upsample_layer.0.": "fourier_last_layer_features.upsample_layer_conv1.",
    "fourier_last_layer_features.upsample_layer.1.": "fourier_last_layer_features.upsample_layer_LayerNorm2d.",
    "fourier_last_layer_features.upsample_layer.3.": "fourier_last_layer_features.upsample_layer_conv2.",
    "fourier_mask_features.downsample_layer.0.": "fourier_mask_features.downsample_layer_conv1.",
    "fourier_mask_features.downsample_layer.1.": "fourier_mask_features.downsample_layer_LayerNorm2d.",
    "fourier_mask_features.downsample_layer.3.": "fourier_mask_features.downsample_layer_conv2.",
    "fourier_first_layer_features.upsample_layer.0.": "fourier_first_layer_features.upsample_layer_conv1.",
    "fourier_first_layer_features.upsample_layer.1.": "fourier_first_layer_features.upsample_layer_LayerNorm2d.",
    "fourier_first_layer_features.upsample_layer.3.": "fourier_first_layer_features.upsample_layer_conv2.",
    "fourier_first_layer_features.dnc_block_combined.SEMBlock.maxpool_conv.selector.fc.0.": "fourier_first_layer_features.dnc_block_combined.SEMBlock.maxpool_conv.selector.fc.",
    "mask_decoder.custom_token_block.mlp.0.": "mask_decoder.custom_token_block.mlp_fc1.",
    "mask_decoder.custom_token_block.mlp.2.": "mask_decoder.custom_token_block.mlp_fc2.",
    "fourier_last_layer_features.dnc_block_combined.SEMBlock.maxpool_conv.selector.fc.0.": "fourier_last_layer_features.dnc_block_combined.SEMBlock.maxpool_conv.selector.fc.",
    "fourier_mask_features.dnc_block_combined.SEMBlock.maxpool_conv.selector.fc.0.": "fourier_mask_features.dnc_block_combined.SEMBlock.maxpool_conv.selector.fc.",
    "mask_decoder.robust_mlp.layers.0.": "mask_decoder.robust_mlp.proj_in.",
    "mask_decoder.robust_mlp.layers.1.": "mask_decoder.robust_mlp.layers.0.",
    "mask_decoder.robust_mlp.layers.2.": "mask_decoder.robust_mlp.proj_out.",
}

# Copied from transformers.models.sam.convert_sam_to_hf.replace_keys
def replace_keys(state_dict):
    model_state_dict = {}
    state_dict.pop("pixel_mean", None)
    state_dict.pop("pixel_std", None)

    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)


        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")


        model_state_dict[key] = value


    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]

    return model_state_dict

# Copied from transformers.models.sam.convert_sam_to_hf.convert_sam_checkpoint with Sam->RobustSam
def convert_robustsam_checkpoint(model_name, checkpoint_path, pytorch_dump_folder, push_to_hub):
    config = get_config(model_name)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    state_dict = replace_keys(state_dict)

    image_processor = RobustSamImageProcessor()
    processor = RobustSamProcessor(image_processor=image_processor)
    hf_model = RobustSamModel(config)
    hf_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_keys = list(hf_model.state_dict().keys())
    state_dict_keys = list(state_dict.keys())

    import json
    import ipdb
    # model_keys = set(hf_model.state_dict().keys())
    # state_dict_keys = set(state_dict.keys())
    # print(len(model_keys),len(state_dict_keys))
    # # 找出存在於 model_keys 但不存在於 state_dict_keys 的鍵
    # missing_in_state_dict = model_keys - state_dict_keys

    # # 找出存在於 state_dict_keys 但不存在於 model_keys 的鍵
    # unexpected_in_model = state_dict_keys - model_keys
    # # 打印不匹配的鍵
    # print("Keys in model but not in state_dict:", missing_in_state_dict)
    # print("Keys in state_dict but not in model:", unexpected_in_model)
    # ipdb.set_trace()

    # # 將合併後的字典保存為 JSON 文件
    # with open('origional.json', 'w') as json_file:
    #     json.dump(state_dict_keys, json_file, indent=4)  # indent=4 會增加空格，方便查看

    # # 將合併後的字典保存為 JSON 文件
    # with open('hf_model.json', 'w') as json_file:
    #     json.dump(model_keys, json_file, indent=4)  # indent=4 會增加空格，方便查看
    # ipdb.set_trace()

    hf_model.load_state_dict(state_dict)
    # load_result = hf_model.load_state_dict(state_dict)
    # assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"
    # assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    # print("\n\n")
    # if len(load_result.missing_keys) != 0 or len(load_result.unexpected_keys) != 0 :
    #     print("Error!!!")
    # else:
    #     print("successful!!")
    # ipdb.set_trace()

    hf_model = hf_model.to(device)

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    input_points = [[[500, 375]]]
    input_labels = [[1]]

    inputs = processor(images=np.array(raw_image), return_tensors="pt").to(device)

    with torch.no_grad():
        output = hf_model(**inputs)
    scores = output.iou_scores.squeeze()

    if model_name == "robustsam_vit_b":
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = hf_model(**inputs)
            scores = output.iou_scores.squeeze()

    elif model_name == "robustsam_vit_h":
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output.iou_scores.squeeze()

        assert scores[-1].item() == 0.9712603092193604

        input_boxes = ((75, 275, 1725, 850),)

        inputs = processor(images=np.array(raw_image), input_boxes=input_boxes, return_tensors="pt").to(device)

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output.iou_scores.squeeze()

        assert scores[-1].item() == 0.8686015605926514

        # Test with 2 points and 1 image.
        input_points = [[[400, 650], [800, 650]]]
        input_labels = [[1, 1]]

        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output.iou_scores.squeeze()

        assert scores[-1].item() == 0.9936047792434692

    if pytorch_dump_folder is not None:
        processor.save_pretrained(pytorch_dump_folder)
        hf_model.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        # repo_id = f"leolu030066/{model_name}" if "slimsam" in model_name else f"leolu030066/{model_name}"
        repo_id = f"leolu030066/{model_name}"
        processor.push_to_hub(repo_id)
        hf_model.push_to_hub(repo_id)
        print("Push to repo !!!")

# Copied from transformers.models.sam.convert_sam_to_hf.__main__ with Sam->RobustSam
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["robustsam_vit_b", "robustsam_vit_h", "robustsam_vit_l"]
    parser.add_argument(
        "--model_name",
        default="robustsam_vit_h",
        choices=choices,
        type=str,
        help="Name of the original model to convert",
    )
    parser.add_argument(
        "--checkpoint_path",
        default = "./robustsam_checkpoint_h.pth",
        type=str,
        required=False,
        help="Path to the original checkpoint",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        default = True,
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    # if "slimsam" in args.model_name:
    #     checkpoint_path = args.checkpoint_path
    #     if checkpoint_path is None:
    #         raise ValueError("You need to provide a checkpoint path for SlimSAM models.")
    # else:
    #     # checkpoint_path = hf_hub_download("ybelkada/segment-anything", f"checkpoints/{args.model_name}.pth")
    #     pass

    convert_robustsam_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
