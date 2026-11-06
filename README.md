# MDFformer: A Multiscale Dual-Modal Fusion Transformer for Semantic Segmentation
Remote sensing image interpretation is a crucial process for Earth observation and geoscience research, and multimodal fusion techniques are key to improving semantic segmentation accuracy. CNN-based methods, due to limited receptive field, suffer from inadequate long-range contextual modeling. While Transformer-based architectures alleviate the limitation, existing multimodal fusion Transformers still face challenges in efficiently integrating multiscale information. Therefore, this work proposes a multiscale dual-modal fusion Transformer, namely MDFformer, to integrate RGB and digital surface model (DSM) cross-modal interaction information. More specifically, MDFformer is a dual-stream encoder architecture combined with a multiscale fusion FPN decoder. MDFformer introduces two novel modules: cross-modality shuffle fusion module (CSFM) and spatial detail enhance module (SDEM), along with deep supervision (DS) to alter the gradient flow and enhance multiscale feature learning. As a result, MDFformer efficiently utilizes both shallow and deep features from diverse modalities across multiple scales. The validation is performed on two publicly available high-resolution datasets, Vaihingen and Potsdam, achieving an overall accuracy (OA) of 92.06% and 91.25%, respectively. Therefore, the model effectively handles RGB and DSM data, making it suitable for remote sensing tasks and efficient earth sensing.




## Usage
Run the code by: python train.py

Test your model and draw the heatmaps by: python test_heatmap.py

Please cite our paper if you find it is useful for your research.

```
@ARTICLE{11172340,
  author={Shi, Leixiong and Dong, Yifei and Wang, Tianqi and Cao, Rui and Wen, Xin},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={MDFformer: A Multiscale Dual-Modal Fusion Transformer for Semantic Segmentation}, 
  year={2025},
  volume={22},
  number={},
  pages={1-5},
  keywords={Transformers;Feature extraction;Training;Data models;Convolution;Remote sensing;Accuracy;Data mining;Artificial intelligence;Semantics;Digital surface model (DSM);multimodality;multiscale;remote sensing;semantic segmentation;Transformer},
  doi={10.1109/LGRS.2025.3611465}}
  ```
