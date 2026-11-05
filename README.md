# STSG & STS-Agri

Research code for **STSG: A Short Text Semantic Graph Model for Similarity Computing Based on Dependency Parsing and Pre-trained Language Models**, with **STS-Agri**, our agricultural sentence-pair similarity dataset.

[Paper](https://doi.org/10.1080/08839514.2024.2321552) · [BibTeX](references.bib) · [STS-Agri dataset](dataset/STS-Agri/STS-Agri.csv)

## Contents

- `dpsimilarity/`: STSG models, dependency parsing, graph attention, and training scripts.
- `baseline/`: pretrained language model baselines.
- `dataset/STS-Agri/STS-Agri.csv`: 200 sentence pairs with `Question1`, `Question2`, binary `Similarity` (0/1), and `Category` columns.
- `pre-models/` and `output/`: local model and output directories.

Other datasets, pretrained weights, and experiment outputs are excluded. Empty directories are preserved with `.gitkeep`. Original source names, including `DPSimilarity` and `biossess`, are retained.

## Setup

Use Linux, Python 3.8, and an NVIDIA GPU. Install a CUDA-compatible PyTorch build for your machine.

```bash
git clone https://github.com/iliaohai/STS-Agri.git
cd STS-Agri
conda create -n stsg python=3.8 -y
conda activate stsg
python -m pip install torch==2.0.1
python -m pip install -r requirements.txt
python -m pip check
```

The curated dependencies are a starting point, not a fully validated environment. `requirements-original.txt` preserves the original environment table and is not a pip requirements file. The code uses legacy `paddle.fluid`; Paddle 3.x is incompatible.

## Prepare models

Place DeBERTa-v3-base model and tokenizer files in `pre-models/deberta-v3-base/`, including `config.json`, `pytorch_model.bin`, `spm.model`, and tokenizer configuration. Obtain weights from the model publisher under its terms. DDParser may download its own resources on first use.

Optional experiments require local data:

| Task | Files under `dataset/` | Required columns |
| --- | --- | --- |
| MRPC | `mrpc/{train,dev,test}.csv` | `#1 String`, `#2 String`, `Quality` |
| BIOSSES | `biosses/{train,dev,test}.tsv` | `sentence1`, `sentence2`, `score` |

## Run STS-Agri

Run from the correct working directory because paths are relative:

```bash
# From the repository root, using bash
export STSG_ROOT="$PWD"
export PYTHONPATH="$STSG_ROOT:$STSG_ROOT/dpsimilarity${PYTHONPATH:+:$PYTHONPATH}"
cd dpsimilarity
python run_STS-Agri.py
# Optional: python run_mrpc.py
# Optional: python run_biossess.py
```

The STS-Agri script splits the data approximately 70/20/10 with seed 42. Defaults are batch size 32, 5,000 epochs, and AdamW learning rate `2e-5`; reduce epochs for an initial check. Logs and validation predictions are written to `output/dpsimilarity/`. The prediction filename is historically `run_mrpc_result.csv`.

## Citation

```bibtex
@article{liao2024stsg,
  title = {{STSG}: A Short Text Semantic Graph Model for Similarity Computing Based on Dependency Parsing and Pre-trained Language Models},
  author = {Liao, Hai and Liang, Yan and Chen, Song and Xiang, Lingyun and Chang, Zhimin and Xiao, Yun},
  journal = {Applied Artificial Intelligence},
  volume = {38},
  number = {1},
  pages = {2321552},
  year = {2024},
  doi = {10.1080/08839514.2024.2321552},
  url = {https://doi.org/10.1080/08839514.2024.2321552}
}
```

Baidu copyright and Apache-2.0 notices in `graph.py` and `gnn.py` are preserved. No repository-wide license was supplied with the source.
