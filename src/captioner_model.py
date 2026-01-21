from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


@dataclass
class CaptionConfig:
    model_id: str = "Salesforce/blip2-flan-t5-base"
    max_new_tokens: int = 40
    num_beams: int = 3
    temperature: float = 1.0
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None


class Blip2Captioner:
    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg

        if cfg.device is None:
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

        if cfg.dtype is None:
            cfg.dtype = torch.float16 if cfg.device == "cuda" else torch.float32

        self.processor = Blip2Processor.from_pretrained(cfg.model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            device_map="auto" if cfg.device == "cuda" else None,
        )

        if cfg.device != "cuda":
            self.model.to(cfg.device)

        self.model.eval()

    @torch.inference_mode()
    def caption(self, image: Image.Image, style: str = "factual", user_prompt: Optional[str] = None) -> str:
        # Qui "style" è tenuto per compatibilità, ma lo stile vero lo passi via user_prompt
        prompt = (user_prompt or "Describe the image.").strip()

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            num_beams=self.cfg.num_beams,
            temperature=self.cfg.temperature,
        )
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
