# MDFformer
Remote sensing image interpretation is a crucial technology for Earth observation and geoscience research, and multimodal fusion techniques are key to improving semantic segmentation accuracy. 

CNN-based methods, due to the limited receptive field, suffer from inadequate long-range contextual modeling. While Transformer-based methods focus on various aspects and effectively improve accuracy, they are not efficient when han-dling multi-scale information. 

Therefore, this work proposes a multi-scale dual-modal fusion approach based on Seg-former,namely MDFformer,to integrate RGB and DSM cross-modal interaction information.

More specifcally, MDFformer is a dual-stream encoder architecture integrated with a multi-scale fusion FPN decoder. We introduce two novel modules: CSFM (Cross-Modality Shuffle Fusion Module) and SDEM (Spatial Detail Enhance Module), along with deep supervision(DS) to alter the gradient flow and enhance multi-scale feature learning. As a result, MDFformer efficiently utilizes both shallow and deep features from diverse modalities across multiple scales. 

We validate the model on two publicly available high-resolution datasets, Vaihingen and Potsdam, achieving an overall accuracy (OA) of 92.04% and 91.2%, respectively.Therefore, the model effectively handles RGB and DSM data, making it suitable for remote sensing tasks and efficient earth sensing.
