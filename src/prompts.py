import yaml
from pathlib import Path

#→ legge e restituisce cfg["styles"]
def load_styles(config_path: str = "configs/caption_styles.yaml") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return cfg["styles"]

#prende la stringa di stile (ex technical) e la restituisce 
def build_style_prompt(style_key: str, styles: dict) -> str:
    if style_key not in styles:
        raise ValueError(f"Unknown style '{style_key}'. Available: {list(styles.keys())}")
    return styles[style_key].strip()
