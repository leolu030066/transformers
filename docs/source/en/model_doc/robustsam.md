<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# RobustSAM

<a href="https://colab.research.google.com/drive/1mrOjUNFrfZ2vuTnWrfl9ebAQov3a9S6E?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![Huggingfaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/robustsam/robustsam/tree/main)

Official repository for RobustSAM: Segment Anything Robustly on Degraded Images

[Project Page](https://robustsam.github.io/) | [Paper](https://arxiv.org/abs/2406.09627) | [Video](https://www.youtube.com/watch?v=Awukqkbs6zM) | [Dataset](https://huggingface.co/robustsam/robustsam/tree/main/dataset)


## Introduction
Segment Anything Model (SAM) has emerged as a transformative approach in image segmentation, acclaimed for its robust zero-shot segmentation capabilities and flexible prompting system. Nonetheless, its performance is challenged by images with degraded quality. Addressing this limitation, we propose the Robust Segment Anything Model (RobustSAM), which enhances SAM's performance on low-quality images while preserving its promptability and zero-shot generalization.

Our method leverages the pre-trained SAM model with only marginal parameter increments and computational requirements. The additional parameters of RobustSAM can be optimized within 30 hours on eight GPUs, demonstrating its feasibility and practicality for typical research laboratories. We also introduce the Robust-Seg dataset, a collection of 688K image-mask pairs with different degradations designed to train and evaluate our model optimally. Extensive experiments across various segmentation tasks and datasets confirm RobustSAM's superior performance, especially under zero-shot conditions, underscoring its potential for extensive real-world application. Additionally, our method has been shown to effectively improve the performance of SAM-based downstream tasks such as single image dehazing and deblurring.


# Model Details

The RobustSAM model is made up of 3 modules:
  - The `VisionEncoder`: a VIT based image encoder. It computes the image embeddings using attention on patches of the image. Relative Positional Embedding is used.
  - The `PromptEncoder`: generates embeddings for points and bounding boxes
  - The `MaskDecoder`: a two-ways transformer which performs cross attention between the image embedding and the point embeddings (->) and between the point embeddings and the image embeddings. The outputs are fed
  - The `Neck`: predicts the output masks based on the contextualized masks produced by the `MaskDecoder`.
# Usage

Below is an example on how to run mask generation given an image and a 2D point:

```python
import torch
from PIL import Image
import requests
from transformers import RobustSamModel, RobustSamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RobustSamModel.from_pretrained("leolu030066/robustsam-vit-huge").to(device)
processor = RobustSamProcessor.from_pretrained("leolu030066/robustsam-vit-huge")

img_url = "https://huggingface.co/leolu030066/robustsam-vit-base/resolve/main/demo/demo_images/blur.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[131, 233], [186, 84], [266, 54]]] # 2D location of a window in the image
input_labels = [[1,1,1]] 

inputs = processor(images=np.array(raw_image), input_points=input_points, input_labels=input_labels,return_tensors="pt").to(device)
with torch.no_grad():
    output = model(multimask_output=False, return_logits=False,**inputs)
    # output = hf_model(multimask_output=False, return_logits=False,clear = True,**inputs)

masks = processor.image_processor.post_process_masks(
    output.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = output.iou_scores
```


## Reference
If you find this work useful, please consider citing us!
```python
@inproceedings{chen2024robustsam,
  title={RobustSAM: Segment Anything Robustly on Degraded Images},
  author={Chen, Wei-Ting and Vong, Yu-Jiet and Kuo, Sy-Yen and Ma, Sizhou and Wang, Jian},
  journal={CVPR},
  year={2024}
}
```


## Acknowledgements
We thank the authors of [SAM](https://github.com/facebookresearch/segment-anything) from which our repo is based off of.
## RobustSamConfig

[[autodoc]] RobustSamConfig

## RobustSamVisionConfig

[[autodoc]] RobustSamVisionConfig

## RobustSamMaskDecoderConfig

[[autodoc]] RobustSamMaskDecoderConfig

## RobustSamPromptEncoderConfig

[[autodoc]] RobustSamPromptEncoderConfig


## RobustSamProcessor

[[autodoc]] RobustSamProcessor


## RobustSamImageProcessor

[[autodoc]] RobustSamImageProcessor


## RobustSamModel

[[autodoc]] RobustSamModel
    - forward


## TFRobustSamModel

[[autodoc]] TFRobustSamModel
    - call
