from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIR / "zebra.jpg"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def resolve_local_path(path: str | Path, base_dir: Path = PROJECT_DIR) -> Path:
    """Resolve a path relative to this file while preserving absolute paths."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def load_image(
    image_path: str | Path = IMAGE_PATH,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Load a local image as an RGB NumPy array.

    Relative paths are resolved beside this file, so ``load_image("zebra.jpg")``
    works even when the script is launched from another working directory.
    With the default ``normalize=True``, the result is ``float32`` in [0, 1].
    Set ``normalize=False`` to keep ``uint8`` values in [0, 255].
    """
    resolved_path = resolve_local_path(image_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Image file was not found: {resolved_path}")

    with Image.open(resolved_path) as source_image:
        rgb_image = source_image.convert("RGB")
        image_array = np.asarray(rgb_image)

    if normalize:
        return image_array.astype(np.float32) / 255.0
    return image_array.copy()


def save_image(image: np.ndarray, image_path: str | Path) -> None:
    """Save an image array to a local PNG/JPEG path."""
    resolved_path = resolve_local_path(image_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    image_array = np.asarray(image)
    if np.issubdtype(image_array.dtype, np.floating):
        if image_array.size and float(np.nanmax(image_array)) <= 1.0:
            image_array = image_array * 255.0
        image_array = np.clip(image_array, 0.0, 255.0).astype(np.uint8)
    elif image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    Image.fromarray(image_array).save(resolved_path)


def cross_correlation_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2-D kernel without flipping it.

    Preserve the input height and width.
    """
    h, w = image.shape[:2]
    kernel_size = kernel.shape[0]
    k = kernel_size // 2
    pad_width = ((k, k), (k, k), (0, 0))
    # padded = np.pad(image, pad_width, mode = "constant", constant_values = 0)
    padded = np.pad(image, pad_width, mode="reflect")
    output = np.zeros(image.shape, dtype = np.float64)
    for x in range(kernel_size):
        for y in range(kernel_size):
            output += kernel[x][y] * padded[x : x + h, y : y + w, ...]

    return output


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image by flipping both kernel axes first."""
    return cross_correlation_2d(image, np.flip(kernel, axis = (0, 1)))

def gaussian_blur_kernel_2d(kernel_size: int, sigma: float) -> np.ndarray:
    """Construct and normalize an odd-sized 2-D Gaussian kernel."""
    k = kernel_size // 2
    output = np.zeros((kernel_size, kernel_size), dtype = np.float64)
    for i in range(kernel_size):
        for j in range(kernel_size):
            output[i][j] = 1 / (2 * np.pi * sigma * sigma) * np.exp(-((i - k) * (i - k) + (j - k) * (j - k) ) / (2 * sigma * sigma)) 
    sum = np.sum(output)
    output /= sum 
    return output



def low_pass(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Blur an image with a Gaussian kernel and your convolution function."""
    kernel = gaussian_blur_kernel_2d(kernel_size, sigma)
    return convolve_2d(image, kernel)


def image_subsampling(image: np.ndarray) -> np.ndarray:
    """Downsample an image by two in each spatial dimension."""
    h, w = image.shape[:2]
    new_h = h // 2
    new_w = w // 2
    output = np.zeros((new_h, new_w, 3), dtype = np.float64)
    for i in range(new_h):
        for j in range(new_w):
            for channel in range(3):
                output[i][j][channel] = image[i << 1][j << 1][channel]
    return output


def gaussian_pyramid(
    image: np.ndarray,
    levels: int = 4,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> list[np.ndarray]:
    """Build a filtered pyramid: original, 1/2, 1/4, and 1/8 levels."""
    output = [image]
    for i in range(levels - 1):
        image = low_pass(image, kernel_size, sigma)
        image = image_subsampling(image)
        output.append(image)
    return output


def direct_pyramid(image: np.ndarray, levels: int = 4) -> list[np.ndarray]:
    """Build the comparison pyramid using direct subsampling only."""
    output = [image]
    for i in range(levels - 1):
        image = image_subsampling(image)
        output.append(image)
    return output


def save_pyramid(
    pyramid: Sequence[np.ndarray],
    output_dir: str | Path = OUTPUT_DIR,
    prefix: str = "pyramid",
) -> None:
    """Save every pyramid level as ``prefix_level_<n>.png``."""
    destination_dir = resolve_local_path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for level_index, level_image in enumerate(pyramid):
        save_image(level_image, destination_dir / f"{prefix}_level_{level_index}.png")


def main() -> None:
    image = load_image("zebra.jpg")

    gaussian = gaussian_pyramid(image, levels=4)
    direct = direct_pyramid(image, levels=4)

    save_pyramid(gaussian, OUTPUT_DIR, "gaussian")
    save_pyramid(direct, OUTPUT_DIR, "direct")

    fig, axes = plt.subplots(
        2, len(gaussian),
        figsize=(16, 7),
        squeeze=False,
    )

    titles = ["原图", "1/2", "1/4", "1/8"]

    for col in range(len(gaussian)):
        axes[0, col].imshow(np.clip(gaussian[col], 0, 1))
        axes[0, col].set_title(titles[col])
        axes[0, col].axis("off")

        axes[1, col].imshow(np.clip(direct[col], 0, 1))
        axes[1, col].set_title(titles[col])
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Gaussian pyramid", fontsize=12)
    axes[1, 0].set_ylabel("Direct subsampling", fontsize=12)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "pyramid_comparison.png", dpi=200)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
