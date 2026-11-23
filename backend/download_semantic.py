"""Explicitly download the pinned local encoder artifacts; never accesses an API key."""
import hashlib
import json
import shutil
from urllib.request import Request, urlopen

from .semantic import FILES, MODEL_REPO, MODEL_REVISION, SiglipEncoder, model_directory, normalized


def valid(path, digest):
    if not path.is_file():
        return False
    if digest is not None:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest() == digest
    try:
        with path.open(encoding="utf-8") as source:
            return "model" in json.load(source)
    except (ValueError, OSError):
        return False


def download():
    directory = model_directory()
    for name, digest in FILES.items():
        target = directory / name
        if valid(target, digest):
            print(f"Ready: {target}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if not valid(part, digest):
            request = Request(
                f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}",
                headers={"Range": f"bytes={offset}-"} if offset else {},
            )
            print(f"Downloading {name} (local offset: {offset:,} bytes)", flush=True)
            with urlopen(request, timeout=90) as response:
                append = response.status == 206 and offset > 0
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError(f"Unexpected download range for {name}")
                with part.open("ab" if append else "wb") as output:
                    shutil.copyfileobj(response, output, length=512 * 1024)
        if not valid(part, digest):
            raise RuntimeError(f"Invalid download for {name}; remove {part} before a clean retry.")
        part.replace(target)
        print(f"Ready: {target}", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", metavar="JPEG_OR_PNG", help="Run both local encoders without downloading")
    args = parser.parse_args()
    if args.check:
        encoder = SiglipEncoder()
        image = normalized(encoder.encode_image(args.check))
        for phrase in ("a room with a television", "une piece avec une television"):
            query = normalized(encoder.encode_text(phrase))
            print(f"Vector dimensions: {image.size}; {phrase!r}: cosine {float(image @ query):.4f}")
    else:
        download()


if __name__ == "__main__":
    main()
