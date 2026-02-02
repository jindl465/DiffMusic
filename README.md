# DiffMusic
[Official Implementation] DiffMusic: Efficient Music Generation From a Single Image Using Diffusion-Based Representations

Authors: Jin Hong, Juhyeon Park, and Junseok Kwon.

Accepted to IEEE Transactions on Audio, Speech, and Language Processing.

## 📖 Overview
DiffMusic is a novel methodology for generating high-quality music from a single static image without relying on expensive Large Language Models (LLMs) or complex multi-modal inputs. Unlike traditional image captioning, DiffMusic extracts genre-related, melodic, and rhythmic elements to bridge the gap between visual and auditory modalities.

## 🚀 Getting Started
**Installation**
```
# Clone the repository
git clone https://github.com/jindl465/DiffMusic.git
cd DiffMusic
# Install dependencies
pip install -r requirements.txt
```

**Inference**
```
python LaDiC/muimage_eval.py --image_path "sample.jpg" --output_dir "./results"
```

## 📝 Citation
If you find this work useful for your research, please cite:

```
@article{hong2026diffmusic,
  title={DiffMusic: Efficient Music Generation From a Single Image Using Diffusion-Based Representations},
  author={Hong, Jin and Park, Juhyeon and Kwon, Junseok},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  year={2026},
  publisher={IEEE}
}
```
