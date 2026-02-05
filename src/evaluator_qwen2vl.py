import torch
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class QwenFidelityEvaluator:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)

        # device_map="auto" gestisce GPU/CPU in automatico
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    @torch.inference_mode()
    def evaluate(self, pil_image, caption: str) -> str:
        prompt = (
            "You are a strict image-caption evaluator.\n"
            "Given an image and a candidate caption:\n"
            "1) List visible objects and key attributes.\n"
            "2) Identify any unsupported claims in the caption (hallucinations).\n"
            "3) Give a fidelity score from 0 to 5.\n"
            "4) Return a short JSON-like report with keys: visible, hallucinations, score, explanation.\n\n"
            f"Caption: {caption}\n"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        # Alcuni tensor vanno su device, ma con device_map auto spesso va bene così.
        gen = self.model.generate(**inputs, max_new_tokens=300)
        out = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return out
