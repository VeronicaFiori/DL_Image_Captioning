import torch

class LavisCaptioner:
    """
    Wrapper per LAVIS InstructBLIP/BLIP2.
    Usa UN SOLO modello (quello scelto qui sotto).
    """

    def __init__(self, device=None, model_name="instructblip"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Import qui per evitare crash se lavis non è installato
        from lavis.models import load_model_and_preprocess

        # Scelta modello: (puoi cambiarla se vuoi un altro checkpoint lavis)
        # Tipico instructblip: instructblip_vicuna7b
        # Se il tuo pc non regge vicuna7b, prova blip2 flan-t5 (più leggero in alcuni casi)
        if model_name == "instructblip":
            name = "instructblip_vicuna7b"
            model_type = "vicuna7b"
        elif model_name == "blip2":
            name = "blip2_t5"
            model_type = "pretrain_flant5xl"
        else:
            raise ValueError("model_name must be 'instructblip' or 'blip2'")

        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=self.device,
        )

    @torch.inference_mode()
    def generate(
        self,
        pil_image,
        instruction: str,
        max_new_tokens: int = 40,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        image = self.vis_processors["eval"](pil_image).unsqueeze(0).to(self.device)

        # prompt “instruction-following”
        prompt = (
            f"Instruction: {instruction}\n"
            f"Task: Write one caption describing the image.\n"
        )

        # LAVIS generate interface cambia tra modelli; questo pattern funziona per molti.
        # Se ti dà errore su generate args, dimmelo e lo adatto al checkpoint specifico.
        out = self.model.generate(
            {"image": image, "prompt": prompt},
            max_length=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return out[0]
