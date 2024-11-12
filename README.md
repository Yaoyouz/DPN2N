# DPneighbor2neighbor:A Self-Supervised Method for Seismic Data Denoising and Key Information Preservation

**Kewen Li**, **Chunlong Li**, **Yuan Xiao**, **Xinyuan Zhu**, **Guangyue Zhou**

**Abstract**:
The data denoising method based on deep learning has become the de facto mainstream denoising algorithm in multiple fields. 
To train an effective denoising deep learning network, an ideal approach is to leverage clean data as labels in a supervised learning setup, provided that both clean and field-noisy data are simultaneously available. Obtaining a significant amount of clean-noise seismic data pairs directly is costly.
Therefore, this study aims to achieve seismic data denoising using the Neighbor2Neighbor self-supervised training paradigm. We propose a novel regularization method for Neighbor2Neighbor that can suppress noise while preserving high-frequency key information such as faults.
We adopt the ResUNet++ network architecture and adjust it based on the frequency domain characteristics of seismic data to enhance the model's information extraction capability and ensure its good generalization performance. To verify the robustness of the model, we simulated various levels of noise on the field data. The experimental results demonstrate that our proposed method achieves effective denoising while preserving key information solely using field noisy data. This approach not only avoids the high cost associated with acquiring a large number of noise-free samples but also effectively mitigates the problem of model performance degradation due to significant differences in the distribution of training and testing data. It can be widely applied to provide high-quality data for subsequent seismic interpretation tasks.


## Python Requirements

This code was tested on:

- Python 3.8
- Pytorch 1.10.0

## Preparing Training Dataset

The data used in this study is publicly available and can be downloaded from [SEG WIKI](https://wiki.seg.org/wiki/Open_data),such as Netherlands F3, New Zeeland Kerry, Poseidon and other data. 

## Repository Structure
- `core` Contains the model architectures used in this project.
- `dataset` The dataloader class is saved here.
- `logdir` The TensorBoard log files and checkpoints are saved here.
- `mydataset` The dataset class is saved here.
- `utils` The utility class is saved here.
- `quick-test.py` Test code.
- `train` Train code.

[//]: # (## Citations)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (```)
