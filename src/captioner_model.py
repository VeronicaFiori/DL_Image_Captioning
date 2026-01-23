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
        # BLIP2-FlanT5 lavora bene con schema Q/A
        prompt = f"Question: {user_prompt}\nAnswer:"

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "num_beams": self.cfg.num_beams,
        }

        # sampling solo se beams=1 e temperature>0
        if self.cfg.num_beams == 1 and self.cfg.temperature is not None and self.cfg.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        #  DECODIFICA DELL'OUTPUT, NON DELL'INPUT
        text = self.processor.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        # ripulisci se il modello include "Answer:"
        if text.lower().startswith("answer:"):
            text = text[len("answer:"):].strip()

        #  se per qualche motivo ripete il prompt, taglia tutto ciò che precede "Answer:"
        low = text.lower()
        if "answer:" in low:
            text = text[low.rfind("answer:") + len("answer:"):].strip()

        return text



    @torch.inference_mode()
    def caption_facts_first(self, image: Image.Image, style_text: str, max_new_tokens: int = 60) -> str:
        # salva config attuale
        old_max = self.cfg.max_new_tokens
        old_beams = self.cfg.num_beams
        old_temp = self.cfg.temperature
        old_top_p = self.cfg.top_p

        try:
            # 1) facts: deterministico
            self.cfg.max_new_tokens = 60
            self.cfg.num_beams = 5
            self.cfg.temperature = 0.0
            self.cfg.top_p = 1.0

            facts_prompt = (
                "List only the visible objects and actions in the image.\n"
                "Rules:\n"
                "- No guessing.\n"
                "- No extra objects.\n"
                "- If unsure write 'unknown'.\n"
                "Return 3-8 bullet points.\n"
            )
            facts = self.caption(image=image, user_prompt=facts_prompt)

            # 2) rewrite: deterministico + vincolato ai facts
            self.cfg.max_new_tokens = max_new_tokens
            rewrite_prompt = (
                "Using ONLY the facts below, write ONE caption.\n"
                "Do not add any object not in the facts.\n"
                "One sentence, max 20 words.\n"
                f"Style requirement: {style_text}\n\n"
                f"FACTS:\n{facts}\n\n"
                "Caption:"
            )
            cap = self.caption(image=image, user_prompt=rewrite_prompt)
            return cap

        finally:
            # ripristina config originale
            self.cfg.max_new_tokens = old_max
            self.cfg.num_beams = old_beams
            self.cfg.temperature = old_temp
            self.cfg.top_p = old_top_p
