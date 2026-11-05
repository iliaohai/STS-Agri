# STS-Agri / STSG

本仓库保存 STSG 相关研究代码及 STS-Agri 实验入口，仅发布源码、配置和文档，不包含数据集、预训练模型、训练权重、实验结果或图片。原始源码目录名为 `DPSimilarity`，模型类名保留为 `DPsimilarityModel`。

关联论文：**STSG: A Short Text Semantic Graph Model for Similarity Computing Based on Dependency Parsing and Pre-trained Language Models**。论文发表于 *Applied Artificial Intelligence*, 38(1), 2321552 (2024)。[论文与 DOI](https://doi.org/10.1080/08839514.2024.2321552)。请使用 [references.bib](references.bib) 引用；GitHub 引用信息见 [CITATION.cff](CITATION.cff)。

## 1. 代码结构

```text
baseline/                 BERT、DeBERTa、T5、SpanBERT 等基线
util/                     日志工具
dpsimilarity/
  model.py                DeBERTa 编码器与二分类头
  model_regression.py     回归模型
  dparsing.py              DDParser 依存分析及语义特征融合
  graph.py                 依存关系到邻接矩阵的映射
  gnn.py                   Paddle 图注意力网络
  run_STS-Agri.py           农业句对二分类实验
  run_mrpc.py               MRPC 二分类实验
  run_biossess.py           BIOSSES 回归实验（文件名沿用原始拼写）
  figure/                  原有绘图代码，图片和表格不发布
dataset/                  本地数据目录，仅保留空目录
pre-models/               本地预训练模型目录，仅保留空目录
output/                   本地运行输出目录，仅保留空目录
```

Git 不记录真正的空目录，所以使用 `.gitkeep` 占位。原有目录结构（缓存、编辑器和 notebook 临时目录除外）保留；数据与模型目录中的所有资产均不发布。

## 2. 准备环境

建议使用 Linux、Conda、Python 3.8 和 NVIDIA CUDA GPU。源码包含直接调用 `.cuda()`，未提供 CPU 模式。以下是基于源码整理的旧版依赖安装方案，**尚未在全新环境完成安装和训练验证**，不能视为已经验证的锁定环境。

```bash
git clone https://github.com/iliaohai/STS-Agri.git
cd STS-Agri
conda create -n stsg python=3.8 -y
conda activate stsg
# 按机器驱动选择适配的 PyTorch CUDA 安装源；此处安装源码使用的版本。
python -m pip install torch==2.0.1
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import paddle, ddparser, transformers; print(paddle.__version__, transformers.__version__)"
```

`requirements-original.txt` 是服务器原始环境表，含多列版本、损坏条目和无关包，作为溯源资料保存，**不要直接 `pip install -r requirements-original.txt`**。整理后的 `requirements.txt` 去除无关依赖并调整明显不匹配的版本；Paddle 采用保留 `paddle.fluid` 的旧版本。若缺少对应平台安装包，应使用匹配的 Python/Linux 环境，不能直接升级到 Paddle 3.x。基线脚本可能需要各自额外依赖和模型。

## 3. 在本地准备数据

数据不随仓库分发。请从你有权使用的数据来源取得文件，并放到下列位置；教程只描述字段，不附任何样本数据。

| 实验 | 本地文件 | 必需列 |
| --- | --- | --- |
| STS-Agri | `dataset/STS-Agri/STS-Agri.csv` | `Question1`, `Question2`, `Similarity` |
| MRPC | `dataset/mrpc/train.csv`, `dev.csv`, `test.csv` | `#1 String`, `#2 String`, `Quality` |
| BIOSSES | `dataset/biosses/train.tsv`, `dev.tsv`, `test.tsv` | `sentence1`, `sentence2`, `score` |

CSV 带表头；TSV 带表头并以制表符分隔。二分类标签应为整数 0/1；BIOSSES 使用数值相似度分数。STS-Agri 脚本以 `random_state=42` 将单文件划分为约 70% 训练、20% 验证、10% 测试。当前入口只在验证集计算指标，测试集并未完成独立评估。

## 4. 在本地准备预训练模型

主模型从 `pre-models/deberta-v3-base/` 读取模型和分词器。将与 Transformers `DebertaV2Model` / `DebertaV2Tokenizer` 兼容的 DeBERTa-v3-base 文件放入该目录，至少准备 `config.json`、PyTorch 权重 `pytorch_model.bin`、SentencePiece 文件 `spm.model` 以及相应分词器配置。可从模型发布方取得这些文件，遵守其许可；本仓库不转存权重。

```bash
# 在仓库根目录运行，只测试本地模型资产是否齐全。
python -c "from transformers import DebertaV2Model,DebertaV2Tokenizer; p='pre-models/deberta-v3-base'; DebertaV2Tokenizer.from_pretrained(p,local_files_only=True); DebertaV2Model.from_pretrained(p,local_files_only=True); print('model files OK')"
```

DDParser 首次运行还可能下载自己的解析器资源，请提前准备网络或对应缓存。BERT、SpanBERT 等基线有独立模型路径，运行前检查各模型文件中的 `pretrained_weights`。

## 5. 启动实验

脚本的资源路径相对于**工作目录**解析。主实验需进入 `dpsimilarity`，同时将仓库根目录和当前目录加入 `PYTHONPATH`，兼容原代码混用包导入和同目录导入的写法。

```bash
# 从仓库根目录执行（Linux bash）
export STSG_ROOT="$PWD"
export PYTHONPATH="$STSG_ROOT:$STSG_ROOT/dpsimilarity${PYTHONPATH:+:$PYTHONPATH}"
cd "$STSG_ROOT/dpsimilarity"
python run_STS-Agri.py
# 其他实验独立运行：
# python run_mrpc.py
# python run_biossess.py
```

`run_STS-Agri.py` 默认 `batch_size=32`、`epochs=5000`，AdamW 实际学习率为 `2e-5`。脚本内另有未使用的 `lr=0.01`，修改它不会改变优化器学习率。建议首次只将 `epochs` 改为 1 做连通性检查；训练样本数需至少达到一个 batch，当前代码会丢弃不足整批的尾部样本。

日志写入 `output/dpsimilarity/`。STS-Agri 入口的最优验证预测沿用历史文件名 `run_mrpc_result.csv`，再次运行可能覆盖；现有脚本不自动保存最佳模型权重。

BERT 农业基线示例：

```bash
cd "$STSG_ROOT/baseline/bert"
export PYTHONPATH="$STSG_ROOT:$STSG_ROOT/baseline/bert${PYTHONPATH:+:$PYTHONPATH}"
python run_STS-Agri.py
```

## 6. 原始实现的复现边界

本次发布保留原始算法代码，不将语法检查等同于训练复现成功：

- `dparsing.py` 在每次调用时重新创建 DDParser 和 GAT；GAT 不属于 PyTorch 主模型的已注册参数。
- 编码器输出经 `detach().cpu().numpy()` 进入 Paddle，返回时重新创建 Torch tensor，因此 GAT 分支没有跨框架端到端梯度；不能将其描述为已联合训练的 GAT。
- 句子拼接手动加入 `[CLS]` / `[SEP]`，同时 tokenizer 又设置 `add_special_tokens=True`；依存词与子词映射可能断言失败。
- 源码设置 `max_length=66`，没有显式启用截断；该参数不构成可靠的输入长度保证。
- 绘图脚本含历史实验常量或历史路径，仅作为原始分析代码发布，不代表随附验证结果。

遇到 `No module named util/model/graph` 时检查上述工作目录和 `PYTHONPATH`；缺少文件时检查本地数据和模型；显存不足时先减小 batch。论文结果的严格复现还需要核对原始实验配置、数据划分和上述实现细节。

## 7. 只上传代码并保留空目录

本仓库通过 `.gitignore` 排除 `dataset/`、`pre-models/`、`output/` 的内容，并允许 `.gitkeep`；常见模型、数据和缓存扩展名也被忽略。发布前仍须审阅暂存清单，忽略规则不会自动移除已跟踪资产。

```bash
git status --short
git add -- '*.py' '*.md' '*.txt' '*.bib' '*.cff' .gitignore
# 显式暂存空目录占位文件
find baseline util dpsimilarity dataset pre-models output -name .gitkeep -exec git add -- {} +
git diff --cached --stat
git diff --cached --name-only
# 确认不含数据、权重、日志或敏感信息后提交
git commit -m "Publish STSG code and reproduction guide"
git push origin main
```

Git 不会递归保存目录本身；新增空目录后创建 `.gitkeep` 再暂存。不要使用 `git add -f` 强行加入数据或模型。这里展示的是当前空仓库的首次发布方式；已有仓库应先检查分支和远端更新。

## 8. 引用与代码来源

完整 BibTeX 见 [references.bib](references.bib)，引用键为 `liao2024stsg`，可在 LaTeX 中使用 `\cite{liao2024stsg}`。该文献关联 STSG 方法，STS-Agri 实验入口不应被误写为论文已报告的所有原始实验。

`graph.py`、`gnn.py` 保留了 Baidu 的 Apache-2.0 版权与许可声明。其余代码在源目录中未发现统一 LICENSE，本次未擅自增加整个仓库的许可；使用和再分发时请核对相应权利及第三方许可。
