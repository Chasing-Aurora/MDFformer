# [TGRS 2024] FTransUNet

**创新点1 CSFM**：class ShuffleBlock 和TimeAttention

**创新点2 SDEM**：class FusionModule

**创新点3 深监督**：model return了多个尺度的张量，并在train函数里计算了损失。

This repo is the official implementation of ['A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation'](https://ieeexplore.ieee.org/document/10458980).

![framework]()

## Usage
you can get the pre-trained model here: https://console.cloud.google.com/storage/browser/vit_models
in the folder: vit_models/imagenet21k/

Run the code by: python train.py

Test your model and draw the heatmaps by: python test_heatmap.py

Please cite our paper if you find it is useful for your research.

```
@ARTICLE{ma2024ftransunet,
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On and Liu, Ming},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2024.3373033}
}
  ```
