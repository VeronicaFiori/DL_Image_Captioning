from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


@dataclass
class CaptionConfig:
    model_id: str = "Salesforce/blip2-flan-t5-xl"  # <-- ESISTE
    max_new_tokens: int = 40
    num_beams: int = 3
    temperature: float = 1.0
    top_p: float = 0.9
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None


class Blip2Captioner:
    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg

        if cfg.device is None:
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

        # fp16 su GPU per velocità/memoria
        if cfg.dtype is None:
            cfg.dtype = torch.float16 if cfg.device == "cuda" else torch.float32

        self.processor = Blip2Processor.from_pretrained(cfg.model_id)

        # su GPU usiamo device_map="auto"
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            device_map="auto" if cfg.device == "cuda" else None,
        )

        if cfg.device != "cuda":
            self.model.to(cfg.device)

        self.model.eval()

    @torch.inference_mode()
    def caption(self, image: Image.Image, user_prompt: str) -> str:
        """Generate a caption given an image + instruction prompt."""
        prompt = user_prompt.strip()

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            num_beams=self.cfg.num_beams,
            temperature=self.cfg.temperature,
        )
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def caption_facts_first(
        self,
        image: Image.Image,
        style_text: str,
        max_new_tokens: int = 60,
    ) -> str:
        # 1) facts extraction (deterministica)
        facts_prompt = (
        "List only the visible objects and actions in the image.\n"
        "Rules:\n"
        "- No guessing.\n"
        "- No extra objects.\n"
        "- If unsure write 'unknown'.\n"
        "Return 3-8 bullet points.\n"
        "Answer:"
        )
        facts = self.caption(
            image=image,
            user_prompt=facts_prompt,
            max_new_tokens=60,
            num_beams=5,
            temperature=0.0,
            top_p=1.0,
        )

        # 2) style rewrite constrained by facts
        rewrite_prompt = (
            "Using ONLY the facts below, write ONE caption.\n"
            "Do not add any object not in the facts.\n"
            "One sentence, max 20 words.\n"
            f"Style requirement: {style_text}\n\n"
            f"FACTS:\n{facts}\n\n"
            "Caption:"
        )
        cap = self.caption(
            image=image,
            user_prompt=rewrite_prompt,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            temperature=0.0,
            top_p=1.0,
        )
        return cap