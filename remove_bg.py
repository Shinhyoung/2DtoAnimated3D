import argparse
from pathlib import Path
from typing import Union, List

from PIL import Image
from rembg import new_session, remove

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SESSION = None


def _get_session(model_name: str = "u2net"):
    global _SESSION
    if _SESSION is None:
        _SESSION = new_session(model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _SESSION


def _collect_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in _SUPPORTED_EXTS)
    raise FileNotFoundError(f"Input not found: {input_path}")


def remove_background(
    input_path: Union[str, Path],
    output_dir: Union[str, Path] = "./output/nobg",
    model_name: str = "u2net",
) -> str:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _get_session(model_name)
    files = _collect_inputs(input_path)
    if not files:
        raise ValueError(f"No supported images found in: {input_path}")

    last_output = ""
    for f in files:
        with Image.open(f) as img:
            img = img.convert("RGBA")
            result = remove(img, session=session)
            if result.mode != "RGBA":
                result = result.convert("RGBA")
        out_path = output_dir / f"{f.stem}.png"
        result.save(out_path, format="PNG")
        last_output = str(out_path)
        print(f"[remove_bg] {f.name} -> {out_path}")

    return last_output


def main():
    parser = argparse.ArgumentParser(description="Remove background using rembg (u2net).")
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output", default="./output/nobg", help="Output directory.")
    parser.add_argument("--model", default="u2net", help="rembg model name (default: u2net).")
    args = parser.parse_args()

    remove_background(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
