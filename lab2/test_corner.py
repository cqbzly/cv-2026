import cv2
import matplotlib.pyplot as plt
import homework2 as hw


# 1. 读取图像
img = cv2.imread("Problem2Images/1_1.jpg")

if img is None:
    raise FileNotFoundError("图片读取失败，请检查运行目录和图片路径")


# 2. 计算 Harris 响应
R = hw.harris_response(
    img,
    hw.HARRIS_K,
    hw.HARRIS_WIN
)


# 3. 设置阈值并选择角点
threshold = hw.CORNER_THRESH * R.max()

pixels = hw.corner_selection(
    R,
    threshold,
    hw.CORNER_MIN_DIST
)

print("图像形状：", img.shape)
print("R 的范围：", R.min(), R.max())
print("检测到的角点数：", len(pixels))
print("前 10 个角点：", pixels[:10])


# 4. 可视化
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)

xs = [x for x, y in pixels]
ys = [y for x, y in pixels]

plt.scatter(
    xs,
    ys,
    s=15,
    facecolors="none",
    edgecolors="red",
    linewidths=0.8
)

plt.title(f"Harris corners: {len(pixels)}")
plt.axis("off")
plt.tight_layout()
plt.savefig("corner_result.png", dpi=200)
plt.show()