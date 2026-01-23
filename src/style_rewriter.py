# src/style_rewriter.py
from dataclasses import dataclass
from typing import Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class RewriteConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: Optional[str] = None
    dtype: Optional[str] = None
    max_new_tokens: int = 40
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True


class StyleRewriter:
    def __init__(self, cfg: RewriteConfig):
        self.cfg = cfg

        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.dtype:
            self.torch_dtype = getattr(torch, cfg.dtype)
        else:
            if self.device == "cuda":
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.torch_dtype = torch.float32

        # ✅ SOLO TOKENIZER (testo)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

        # ✅ SOLO MODELLO TESTO
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def rewrite(self, base_caption: str, style_text: str, extra: str = "") -> str:
        system = (
            "You are a careful rewriting assistant.\n"
            "Rewrite the caption WITHOUT changing its meaning.\n"
            "Do NOT add objects, actions, attributes, or counts.\n"
            "Output ONE sentence (max 20 words).\n"
            "Output ONLY the rewritten caption.\n"
        )

        user = (
            f"STYLE REQUIREMENT:\n{style_text}\n\n"
            f"BASE CAPTION:\n{base_caption}\n\n"
            "Rewrite the caption now."
        )

        if extra.strip():
            user += f"\nExtra requirement:\n{extra.strip()}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # prendi ultima riga “pulita”
        out = text.strip().splitlines()[-1]
        out = re.sub(r"\s+", " ", out).strip(" \"'")

        # sicurezza: max 20 parole
        words = out.split()
        if len(words) > 20:
            out = " ".join(words[:20]).rstrip(",.;:") + "."

        return out
