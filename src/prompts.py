import yaml
from pathlib import Path

def load_styles(config_path: str = "configs/caption_styles.yaml") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return cfg["styles"]

def build_style_prompt(style_key: str, styles: dict) -> str:
    if style_key not in styles:
        raise ValueError(f"Unknown style '{style_key}'. Available: {list(styles.keys())}")
    # Instruction-style prompt (InstructBLIP-like)
    return styles[style_key].strip()
