from PIL import Image

#carica un'immagine PIL da file
def load_pil_image(file) -> Image.Image:
    return Image.open(file).convert("RGB")
