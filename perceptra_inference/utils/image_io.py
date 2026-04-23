"""Image I/O utilities — mirrors perceptra-seg/perceptra_seg/utils/image_io.py."""

import io
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from perceptra_inference.exceptions import ImageLoadError


def load_image(
    image: "np.ndarray | Image.Image | bytes | str | Path",
    timeout: int = 10,
) -> np.ndarray:
    """Load image from numpy array, PIL Image, bytes, file path, or URL.

    Returns:
        RGB image as numpy array (HxWx3)
    """
    try:
        if isinstance(image, np.ndarray):
            return _ensure_rgb(image)

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if isinstance(image, bytes):
            pil_img = Image.open(io.BytesIO(image))
            return np.array(pil_img.convert("RGB"))

        if isinstance(image, (str, Path)):
            image_str = str(image)
            if image_str.startswith(("http://", "https://")):
                response = requests.get(image_str, timeout=timeout)
                response.raise_for_status()
                pil_img = Image.open(io.BytesIO(response.content))
                return np.array(pil_img.convert("RGB"))

            path = Path(image_str)
            if not path.exists():
                raise ImageLoadError(f"Image file not found: {path}")
            pil_img = Image.open(path)
            return np.array(pil_img.convert("RGB"))

        raise ImageLoadError(f"Unsupported image type: {type(image)}")

    except ImageLoadError:
        raise
    except Exception as e:
        raise ImageLoadError(f"Failed to load image: {e}") from e


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ImageLoadError(f"Unexpected image shape: {image.shape}")
