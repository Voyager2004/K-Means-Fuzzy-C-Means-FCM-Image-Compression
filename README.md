# K-Means + Fuzzy C-Means (FCM) Image Compression

> **Course-work enhancement of Andrew Ng’s Reinforcement Learning Specialization  
> “K-Means for Image Compression” lab (Coursera, 2023).**  
>  
> This repo re-implements the original lab, then extends it with fuzzy clustering, quality-oriented evaluation metrics, and a cleaner Python API.  
> **Educational use only – not affiliated with DeepLearning.AI, Stanford, or Coursera.**

---

## ✨ Improvements over the original lab

| 功能 | 原始课堂练习 | **本仓库** |
|------|--------------|-----------|
| **硬 K-Means** | ✔️ | ✔️ |
| **Fuzzy C-Means 细化** | ❌ | **✔️** (`fcm_core.py`) — 缓解色块生硬、提升 PSNR/SSIM |
| **误差扩散抖动 (可选)** | ❌ | **✔️**（Notebook 中单元格） |
| **质量评估** | 仅视觉对比 | **PSNR / SSIM 自动计算** |
| **可调超参数** | 固定 `K=16` | `K`, `m`, `max_iter`, `tolerance` 均可在 Notebook 或脚本里指定 |
| **结构整理** | 单一 Notebook | 分离 `utils.py` / `fcm_core.py`，便于脚本化调用 |

---

## 🗂️ Repository Layout

```

.
├── fcm\_core.py                     # FCM 封装（基于 scikit-fuzzy）
├── utils.py                        # 绘图、评价等辅助函数
├── notebooks/
│   └── K-means+FCM的图片压缩方法.ipynb
├── examples/                       # 若干 ≤100 KB 演示图片
├── requirements.txt
└── README.md

````

---

## 🔧 Installation

```bash
# 建议使用虚拟环境
python -m venv .venv && source .venv/bin/activate   # Windows 用户改用 .venv\Scripts\activate
pip install -r requirements.txt
````

---

## 🚀 Quick Start

### 1. Jupyter Notebook（推荐体验）

```bash
jupyter lab
# 打开 notebooks/K-means+FCM的图片压缩方法.ipynb
# 按顺序运行，或修改 K / m / max_iter 等超参数后重新运行
```

运行结束后，Notebook 会同时显示：

* 原图、K-Means 压缩图、K-Means + FCM 图
* 每步迭代的聚类中心可视化
* PSNR / SSIM 定量指标

### 2. 在脚本中批量调用

下面的最小示例展示了如何用仓库里的函数压缩单张图片并保存结果：

```python
import numpy as np
from PIL import Image
from utils import kmeans_compress      # Notebook 中已定义
from fcm_core import run_fcm

# 载入图片并展开为 N×3
img   = Image.open("examples/catking.png").convert("RGB")
X     = np.asarray(img, dtype=np.float32).reshape(-1, 3)

# ① K-Means 压缩
X_km, labels_km, centroids_km = kmeans_compress(X, K=32, max_iter=10)

# ② 可选：FCM 细化
centroids_fcm, _ = run_fcm(X, labels_km, m=2.0, max_iter=8)
X_fcm = centroids_fcm[labels_km]           # 利用 K-Means 标签快速映射

# ③ 保存图片
Image.fromarray(X_km.reshape(img.size[1], img.size[0], 3).astype(np.uint8))\
     .save("catking_kmeans.png")
Image.fromarray(X_fcm.reshape(img.size[1], img.size[0], 3).astype(np.uint8))\
     .save("catking_fcm.png")
```

---

## 🧑‍🎓 Academic Notice

Please cite the original lab when you use or adapt this repository for
research or teaching:

> **Ng, A.** “Reinforcement Learning Specialization”,
> DeepLearning.AI / Stanford University, Coursera, 2023 –
> *Week ? – “K-Means for Image Compression” exercise.*

All original materials are © DeepLearning.AI and provided here under
Coursera’s educational fair-use policy.

---

## 📄 License

MIT License.  See `LICENSE` for details.

---
