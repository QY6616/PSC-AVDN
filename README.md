# Parse, Search, and Confirmation: Training-Free Aerial Vision-and-Dialog Navigation

This repository provides the inference code for our CVPR 2026 paper:

> **Parse, Search, and Confirmation: Training-Free Aerial Vision-and-Dialog Navigation with Chain-of-Thought Reasoning and Structured Spatial Memory**  
> Qi Y, Li H, Huang S, et al. CVPR 2026.

The method is built on the [AVDN](https://github.com/UCSB-AI/Aerial-Vision-and-Dialog-Navigation) benchmark and supports two settings:

- **ANDH**: sub-trajectory, single-step navigation
- **ANDH-Full**: full-trajectory, multi-step navigation

---

## 1. Environment and Dataset

Please follow the official AVDN repository for environment installation and dataset preparation:

https://github.com/UCSB-AI/Aerial-Vision-and-Dialog-Navigation

In summary:

1. Install dependencies (PyTorch, torchvision, and `requirements.txt`) as described in the AVDN repo.
2. Download xView satellite images and place them under `datasets/AVDN/train_images/`.
3. Download AVDN annotation files to `datasets/AVDN/annotations/`.

### Recommended project layout

The inference scripts resolve paths relative to the AVDN project root (the directory that contains `datasets/` and `src/`). We recommend placing this repository under that root:

```text
AVDN-project-root/
├── datasets/
│   └── AVDN/
├── src/
└── PSC-AVDN/
    ├── README.md
    └── script/
        ├── ANDH/
        ├── ANDH-Full/
        ├── src/
        └── util.py
```

For **ANDH-Full**, place full-trajectory annotations under `datasets/FULL/`.

---

## 2. API Configuration

This method uses two external LLM APIs:

- **Parsing**: instruction analysis (`Parsing.py`)
- **Search & Confirmation**: visual grounding (`Search_Confirmation.py`)

Set the following environment variables before running:

```bash
export API_TOKEN="your_token"   # used by Parsing.py
export API_KEY="your_key"       # used by Search_Confirmation.py
```

Also configure the API endpoint and model names in the corresponding scripts:

- `script/ANDH/Parsing.py` and `script/ANDH-Full/Parsing.py`: `CFGPU_URL`, `CFGPU_MODEL`
- `script/ANDH/Search_Confirmation.py` and `script/ANDH-Full/Search_Confirmation.py`: `QWEN_URL`, `QWEN_MODEL`

---

## 3. Inference

Run the commands below from the `PSC-AVDN` directory.

### ANDH (sub-trajectory)

**Step 1. Parse instructions**

```bash
python script/ANDH/Parsing.py
```

Output:

- `preds_out/parsing_results.csv`

**Step 2. Search and confirm destinations**

```bash
python script/ANDH/Search_Confirmation.py
```

Output:

- `preds/andh/search_output/`

Default split: `test_unseen`.

---

### ANDH-Full (full trajectory)

**Step 1. Parse instructions**

```bash
python script/ANDH-Full/Parsing.py
```

Output:

- `preds_out/parsing_results_full.csv`

**Step 2. Search and confirm destinations**

```bash
python script/ANDH-Full/Search_Confirmation.py
```

Output:

- `preds/andh_full/search_output/`

Default split: `val_seen_full`.

You can modify `SPLIT`, input/output paths, and other hyperparameters at the top of each script if needed.

---

## Citation

If you find this work useful, please cite our paper and the AVDN dataset:

```bibtex
@inproceedings{qi2026parse,
  title={Parse, Search, and Confirmation: Training-Free Aerial Vision-and-Dialog Navigation with Chain-of-Thought Reasoning and Structured Spatial Memory},
  author={Qi, Y and Li, H and Huang, S and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23859--23868},
  year={2026}
}

@inproceedings{fan-etal-2023-aerial,
  title={Aerial Vision-and-Dialog Navigation},
  author={Fan, Yue and Chen, Winson and Jiang, Tongzhou and Zhou, Chun and Zhang, Yi and Wang, Xin Eric},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={3043--3061},
  year={2023},
  url={https://aclanthology.org/2023.findings-acl.190},
  doi={10.18653/v1/2023.findings-acl.190}
}
```
