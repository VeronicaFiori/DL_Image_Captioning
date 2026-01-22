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
    device: Optional[str] = None  # "cuda" o "cpu"
    dtype: Optional[str] = None   # "float16" se cuda


class Blip2Captioner:
    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg

        if cfg.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        self.device = device

        # dtype: su GPU conviene float16
        if cfg.dtype is None:
            self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            self.torch_dtype = getattr(torch, cfg.dtype)

        self.processor = Blip2Processor.from_pretrained(cfg.model_id)

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def caption(
        self,
        image: Image.Image,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        # fallback ai default
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        num_beams = num_beams if num_beams is not None else self.cfg.num_beams
        temperature = temperature if temperature is not None else self.cfg.temperature
        top_p = top_p if top_p is not None else self.cfg.top_p

        # BLIP2-FlanT5 lavora bene con istruzioni tipo "Question: ... Answer:"
        prompt = f"Question: {user_prompt}\nAnswer:"

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        # sampling solo se vuoi temperatura > 0 e beam=1 (altrimenti di solito si usa beam search)
        if num_beams == 1 and temperature is not None and temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        #  decodifica dell'OUTPUT (non dell'input!)
        text = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Alcuni modelli possono includere "Answer:" nell'output: lo ripuliamo
        if text.lower().startswith("answer:"):
            text = text[len("answer:"):].strip()

        return text
