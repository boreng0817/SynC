# SynC
**SynC: Synthetic Image Caption Dataset Refinement with One-to-many Mapping for Zero-shot Image Captioning** MM 2025

Si-Woo Kim, MinJu Jeon, Ye-Chan Kim, Soeun Lee, Taewhan Kim, Dong-Jin Kim

Official implementation of **SynC**.

---

### Conda environment
```bash
git clone https://github.com/boreng0817/SynC.git
cd SynC

conda create -n sync python=3.9
conda activate sync
pip install -r requirements.txt
```

### Data preparation
Download annotation files from the GitHub release.

```bash
bash scripts/download.sh
```

This script will:
- download the released annotation assets
- verify file integrity with SHA256
- merge split zip files
- extract the `annotations/` directory

### Quick start
```bash
python main.py
```

## Citation
If you use this code for your research, please cite:

```bibtex
@inproceedings{kim2025sync,
  title={SynC: Synthetic Image Caption Dataset Refinement with One-to-many Mapping for Zero-shot Image Captioning},
  author={Kim, Si-Woo and Jeon, MinJu and Kim, Ye-Chan and Lee, Soeun and Kim, Taewhan and Kim, Dong-Jin},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={2683--2692},
  year={2025}
}
```

## Acknowledgments

Thanks to the authors of prior open-source projects that inspired this repository [PCM-Net](https://jianjieluo.github.io/SynthImgCap/).
