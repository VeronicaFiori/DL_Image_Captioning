# src/style_rewriter.py
from dataclasses import dataclass
from typing import Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class RewriteConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: Optional[str] = None          # "cuda" o "cpu"
    dtype: Optional[str] = None           # "float16" / "bfloat16" / "float32"
    max_new_tokens: int = 40
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True


class StyleRewriter:
    def __init__(self, cfg: RewriteConfig):
        self.cfg = cfg

        if cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device

        if cfg.dtype is None:
            if self.device == "cuda":
                # bf16 se supportato, altrimenti fp16
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = getattr(torch, cfg.dtype)

        self.tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

        # prova 4-bit se c'è bitsandbytes (utile su GPU piccole)
        model_kwargs = dict(torch_dtype=self.torch_dtype)
        try:
            import bitsandbytes  # noqa: F401
            if self.device == "cuda":
                model_kwargs.update(dict(load_in_4bit=True, device_map="auto"))
        except Exception:
            pass

        if "device_map" not in model_kwargs:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)

        self.model.eval()

    @torch.inference_mode()
    def rewrite(self, base_caption: str, style_text: str, extra: str = "") -> str:
        # Prompt molto “hard” anti-invenzioni
        sys = (
            "You are a careful rewriting assistant.\n"
            "You MUST rewrite the caption WITHOUT changing meaning.\n"
            "You MUST NOT add objects, actions, attributes, text, logos, counts.\n"
            "If the style asks for hashtags/technical tone, apply it ONLY by rephrasing.\n"
            "Output ONE sentence (max 20 words). Output ONLY the final caption.\n"
        )

        user = (
            f"STYLE REQUIREMENT:\n{style_text}\n\n"
            f"BASE CAPTION:\n{base_caption}\n\n"
            "TASK:\nRewrite the base caption to match the style.\n"
        )
        if extra.strip():
            user += f"\nEXTRA:\n{extra.strip()}\n"

        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)

        gen = dict(
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=bool(self.cfg.do_sample),
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
        )
        # se vuoi 100% deterministico:
        # gen["do_sample"] = False
        # gen.pop("temperature", None); gen.pop("top_p", None)

        out = self.model.generate(**inputs, **gen)
        text = self.tok.decode(out[0], skip_special_tokens=True)

        # prendiamo l’ultima parte generata: dopo il prompt
        # (best effort: estrai ultima riga “pulita”)
        last = text.splitlines()[-1].strip()
        last = re.sub(r"\s+", " ", last).strip()
        # safety: se ha messo virgolette o roba strana
        last = last.strip('"').strip("'").strip()

        # safety: taglia a ~20 parole se esagera
        words = last.split()
        if len(words) > 20:
            last = " ".join(words[:20]).rstrip(" ,;:") + "."

        return last
