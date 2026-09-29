"""Record docs/images/demo.gif from the demo page.

Needs the service running (`docker compose up`) with a real LLM configured: three
emails, well under 1 US cent with gpt-4.1-mini. Uses the installed Microsoft Edge,
so no browser download is needed.

    uv run python -m scripts.record_demo --url http://localhost:8000/
"""

import argparse
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images" / "demo.gif"
VIEWPORT = {"width": 900, "height": 920}

# (example button, approve the draft?): a return that is approved, one outside the
# return window, and a question about someone else's order.
SCENES = [
    ("Devolución en plazo", True),
    ("Fuera de plazo", False),
    ("Pedido de otra persona", False),
]


class Recorder:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.frames: list[tuple[Image.Image, int]] = []

    def shot(self, ms: int) -> None:
        image = Image.open(io.BytesIO(self.page.screenshot()))
        self.frames.append((image.convert("RGB"), ms))

    def save(self, path: Path) -> None:
        # One palette built from every frame keeps colours right (and stable) and the file small.
        width, height = self.frames[0][0].size
        strip = Image.new("RGB", (width, height * len(self.frames)))
        for i, (frame, _) in enumerate(self.frames):
            strip.paste(frame, (0, height * i))
        palette = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        images = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f, _ in self.frames]
        path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=[ms for _, ms in self.frames],
            loop=0,
            optimize=True,
        )


def record(base_url: str, out: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        rec = Recorder(page)
        for example, approve in SCENES:
            page.goto(base_url)
            page.get_by_role("button", name=example).click()
            rec.shot(1800)
            page.click("#send")
            page.wait_for_selector("#result:not(.hidden)", timeout=120_000)
            rec.shot(5200)
            if approve:
                page.click("#approve")
                page.wait_for_selector("#status.sent", timeout=30_000)
                rec.shot(3000)
        browser.close()
    rec.save(out)
    print(f"{out} · {len(rec.frames)} frames · {out.stat().st_size / 1024:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--url", default="http://localhost:8000/")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    record(args.url, args.out)


if __name__ == "__main__":
    main()
