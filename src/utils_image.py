from PIL import Image

def load_pil_image(file) -> Image.Image:
    return Image.open(file).convert("RGB")
