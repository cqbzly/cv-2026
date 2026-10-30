# cv-2026

2026 年计算机视觉课程实验，按本地最终提交压缩包归档。

| 目录 | 内容 | 最终提交来源 |
| --- | --- | --- |
| [lab1](lab1/) | 图像金字塔代码、实验报告、输入图片 | `HW1/HW1.zip` |
| [lab2](lab2/) | 角点、匹配、RANSAC、图像融合、平面全景和圆柱投影 bonus；含报告与已提交结果 | `output/2500012966_刘洋_HW2.zip` |
| [lab3](lab3/) | 基础矩阵、相机标定、消失点、三角化和无真值匹配；含 PDF 报告与已提交结果 | `output/2500012966_刘洋_HW3.zip` |

所有提交包中的源码均原样保留。另补充原实验目录的输入图片和依赖文件，以便运行。
各文件与提交包的对应关系、SHA-256 见 [SUBMISSION_MANIFEST.json](SUBMISSION_MANIFEST.json)。

## 运行

```sh
cd lab1
python -m pip install -r requirements.txt
python code.py
```

第二次实验的环境和运行命令见 [lab2/README.md](lab2/README.md)。
第三次实验的环境和运行命令见 [lab3/README.md](lab3/README.md)。
