# Parse, Search, and Confirmation: Training-Free Aerial Vision-and-Dialog Navigation

This repository provides the inference code for our CVPR 2026 paper:

> **Parse, Search, and Confirmation: Training-Free Aerial Vision-and-Dialog Navigation with Chain-of-Thought Reasoning and Structured Spatial Memory**  
---

## 1. Environment and Dataset

Please follow the official AVDN repository for environment installation and dataset preparation:

https://github.com/UCSB-AI/Aerial-Vision-and-Dialog-Navigation


### Recommended project layout
```text
AVDN-project-root/
├── datasets/
│   └── AVDN/
├── src/
└── PSC-AVDN/
    └── script/
        ├── ANDH/
        ├── ANDH-Full/
        ├── src/
        └── util.py
```
---
## 2. Inference

Run the commands below from the `PSC-AVDN` directory.

### ANDH (sub-trajectory)

**Step 1. Parse**

```bash
python script/ANDH/Parsing.py
```

**Step 2 and Step 3. Search and confirm destinations**

```bash
python script/ANDH/Search_Confirmation.py
```

---

### ANDH-Full (full trajectory)

**Step 1. Parse**

```bash
python script/ANDH-Full/Parsing.py
```
**Step 2 and Step 3. Search and confirm destinations**

```bash
python script/ANDH-Full/Search_Confirmation.py
```
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
