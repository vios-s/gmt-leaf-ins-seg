# gmt-leaf-ins-seg
Official repository for GMT: Guided Mask Transformer for Leaf Instance Segmentation (https://arxiv.org/abs/2406.17109), accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025 as an oral presentation.

## About the repo
- Install libs using environment.yml
- GMT model architectures are stored at ./mask2former (e.g., guide_xxx.py)
- Method of training guide functions: ./harmonic/get_embeddings.py
  - Trained guide functions for different datasets: ./harmonic/guide_functions
- Configuration files for different datasets: ./configs
- GMT Training code: ./guide_train_net.py
- Results in the paper: ./submission/results_in_paper

## Acknowledgements
We thank the authors of the following repositories for opening source their code:

Mask2Former: https://github.com/facebookresearch/Mask2Former

Harmonic Embeddings: https://github.com/kulikovv/harmonic

DFPQ: https://github.com/ziplab/FASeg
