# HW3 多视图几何实验

刘洋 2500012966

`main.py`、PDF 实验报告和 `outputs/` 原样来自最终提交包 `output/2500012966_刘洋_HW3.zip`，SHA-256 和压缩包条目见仓库根目录的 [SUBMISSION_MANIFEST.json](../SUBMISSION_MANIFEST.json)。另从原实验目录补充 `data/` 和 `requirements.txt`，用于复现实验。

## 内容

- Q1：归一化与非归一化八点法估计基础矩阵，绘制极线。
- Q2：估计相机投影矩阵，分解内外参并计算重投影误差。
- Q3：根据线段和消失点估计焦距。
- Q4：三角化恢复三维点，评估三维与重投影误差。
- Q5：使用 SIFT 匹配和 RANSAC 估计基础矩阵。

`outputs/` 包含可视化 PNG、`metrics.json` 指标和 `matrices.npz` 矩阵。`data/lab1_segments.csv` 保存 Q3 线段，标注来源与假设见 `data/segments_provenance.json`。

## 运行

在 `lab3` 目录中安装依赖并运行：

```sh
python -m pip install -r requirements.txt
python main.py
```

输入数据路径相对于 `main.py`，默认不弹出绘图窗口。重新运行会更新 `outputs/`。若要保留提交时的结果，可以在 `lab3` 目录指定其他输出目录：

```sh
python -c "from main import run_assignment; run_assignment('outputs_rerun')"
```
