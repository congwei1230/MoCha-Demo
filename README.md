# MoCha Demo Implementation

This repository provides a demo implementation of MoCha built on top of HunyuanVideo.

We fine-tune HunyuanVideo on the Hallo3 dataset.
Due to differences in training data, model scale, and training strategy, this demo does not fully reproduce the performance of the original MoCha model, but it reflects the core design and and serves as a baseline for further research and study.


Supported Tasks

This implementation supports two generation modes:

- st2v: speech + text → video
- sti2v: image + speech + text → video

## How to use

### 1. Create Conda Environment

```
conda env create -f environment.yml
conda activate mocha
```

This environment is tested with:
- Python 3.11
- PyTorch 2.4.1 + CUDA 12.1
- diffusers 0.36.0
- transformers 4.49.0


### 2. Download Checkpoint

Download the MoCha transformer checkpoint to a local path:

```
python download_ckpt.py
```

After downloading, record the local path to the checkpoint file (e.g. model.ckpt).


### 3. Inference


Speech + Text → Video (st2v)

```
python inference.py \
  --task st2v \
  --audio_path demos/man_1.mp3 \
  --output_path demos/output.mp4 \
  --transformer_ckpt_path /path/to/your/model.ckpt
```

Speech + Image + Text → Video (sti2v)

```
python inference.py \
  --task sti2v \
  --audio_path demos/man_1.mp3 \
  --i2v_img_path demos/man_1.png \
  --output_path demos/output.mp4 \
  --transformer_ckpt_path /path/to/your/model.ckpt
```


## Citation

If you find this work useful, please give us a free cite:

```
@article{wei2025mocha,
  title={Mocha: Towards movie-grade talking character synthesis},
  author={Wei, Cong and Sun, Bo and Ma, Haoyu and Hou, Ji and Juefei-Xu, Felix and He, Zecheng and Dai, Xiaoliang and Zhang, Luxin and Li, Kunpeng and Hou, Tingbo and others},
  journal={arXiv preprint arXiv:2503.23307},
  year={2025}
}
```