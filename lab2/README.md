# HW2 图像拼接实验

刘洋 2500012966

## 文件与结果

* `homework2.py`、`utils.py`：基础角点、描述子、匹配、DLT、RANSAC、融合及平面全景。
* `cylindrical.py`：圆柱投影、平移 RANSAC、feather 融合、焦距估计及完整一圈闭合。
* `outputs/`：全部已有实验结果，包含图片、单应矩阵、内点集合及统计 JSON。
* `outputs/cylindrical/`：四组五帧和完整一圈的圆柱结果、单图投影与 mask。

所有命令都在解压后的 `HW2` 目录运行，路径不依赖原电脑的盘符。实验代码不依赖 Word、Office 或额外报告库。

## 环境

推荐 Python 3.13，当前实测为 3.13.13。创建环境并安装实验依赖：

```powershell
python -m venv .venv
# Windows PowerShell 不必激活，直接使用环境中的 Python
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

下文的 `python` 代表已经装好依赖的解释器；也可以替换为 `.venv/Scripts/python.exe`。Linux/macOS 对应 `.venv/bin/python`。

图像用 OpenCV 读取，为 BGR 三通道；显示到 Matplotlib 前转 RGB。几何坐标使用 `(x,y)`，NumPy 图像索引使用 `[y,x]`。不要将灰度图重新扩展通道当成恢复彩色，输出应采样彩色原图。

## 基础实验

```powershell
# Q1 角点及可视化，显示窗口需要图形界面
python test_corner.py

# Q2/Q3 第四组描述子匹配
python test_histogram.py

# Q4.2 内点、外点及统计
python test_ransac.py --pair 1_1.jpg 1_2.jpg

# 四组双图融合
python test_blend.py

# 四场景推荐连续五帧的平面全景
python test_panorama.py

# 指定一个场景或窗口
python test_panorama.py --scene library --frames 5 --start 8

```

## 圆柱投影 bonus

```powershell
# 四个场景完整一圈，利用一圈约束估计焦距，首尾闭合
python test_cylindrical.py --frames 0 --close-loop --auto-focal

# 只跑第三个场景
python test_cylindrical.py --scene parrington --frames 0 --close-loop --auto-focal

# 五帧对比，使用本次完整序列估计出的焦距
python test_cylindrical.py --scene grail --frames 5 --focal 627.5606024840334
python test_cylindrical.py --scene library --frames 5 --focal 654.4629437754564
python test_cylindrical.py --scene parrington --frames 5 --focal 702.8390018829886
python test_cylindrical.py --scene Xue-Mountain-Entrance --frames 5 --focal 788.023431538016

# 几何与采样检查
python -m unittest discover -s . -p test_cylindrical_geometry.py -v
```

`--auto-focal` 只适用于首尾重叠且恰好完整一圈的序列，不可用于任意局部窗口。默认五帧不带 `--focal` 时仅猜测 `f=图像宽度`，与本次报告的焦距设置不同。焦距单位为像素，缩放图片时要同比缩放焦距。
