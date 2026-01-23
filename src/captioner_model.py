from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


@dataclass
class CaptionConfig:
    model_id: str = "Salesforce/blip2-flan-t5-xl"
    max_new_tokens: int = 40
    num_beams: int = 3
    temperature: float = 1.0
    top_p: float = 0.9
    device: Optional[str] = None   # "cuda" o "cpu"
    dtype: Optional[str] = None    # "float16" / "float32"


class Blip2Captioner:
    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg

        # --- device ---
        if cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device

        # --- dtype ---
        if cfg.dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            # esempio: cfg.dtype="float16" -> torch.float16
            self.torch_dtype = getattr(torch, cfg.dtype)

        # --- load processor + model ---
        self.processor = Blip2Processor.from_pretrained(cfg.model_id)

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def caption(self, image: Image.Image, user_prompt: str) -> str:
        """
        Genera una caption usando i parametri in self.cfg.
        """
        prompt = f"Question: {user_prompt}\nAnswer:"

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        # sposta tensori su device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": int(self.cfg.max_new_tokens),
            "num_beams": int(self.cfg.num_beams),
        }

        # sampling SOLO se num_beams==1
        if int(self.cfg.num_beams) == 1 and float(self.cfg.temperature) > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=float(self.cfg.temperature),
                top_p=float(self.cfg.top_p),
            )

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.processor.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        # --- pulizia robusta ---
        text = decoded

        # 1) Se il modello ha ripetuto il prompt, rimuovi ESATTAMENTE il prompt (non rfind)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        # 2) Se rimane "Answer:" all'inizio, rimuovilo
        if text.lower().startswith("answer:"):
            text = text[len("answer:"):].strip()

        # 3) Se dopo la pulizia è vuoto, NON buttare tutto: tieni decoded (fallback)
        if not text:
            text = decoded
            if text.lower().startswith("answer:"):
                text = text[len("answer:"):].strip()    
        return text


    @torch.inference_mode()
    def caption_facts_first(self, image: Image.Image, style_text: str, max_new_tokens: int = 60) -> str:
        """
        Two-step:
        1) Estrae facts visibili in modo deterministico
        2) Riscrive UNA caption con lo stile richiesto, vincolata ai facts
        """
        # salva config attuale
        old = (self.cfg.max_new_tokens, self.cfg.num_beams, self.cfg.temperature, self.cfg.top_p)

        try:
            # 1) facts (deterministico)
            self.cfg.max_new_tokens = 80
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

            # 2) rewrite vincolato ai facts (deterministico)
            self.cfg.max_new_tokens = int(max_new_tokens)
            self.cfg.num_beams = 5
            self.cfg.temperature = 0.0
            self.cfg.top_p = 1.0

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
            self.cfg.max_new_tokens, self.cfg.num_beams, self.cfg.temperature, self.cfg.top_p = old
