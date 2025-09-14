#!/usr/bin/env python3
"""
Random-Subset Text Animation -> Animated GIF
Each frame displays a random subset of the characters from the input text.
Author: ChatGPT + humans.  It is very unlikely to turn your computer into a paperclip maximizer.
This code requires the file DejaVuSans.ttf, which you can get from https://dejavu-fonts.github.io/
"""

from PIL import Image, ImageDraw, ImageFont
import random
import argparse
import os
from typing import List, Tuple

def _coerce_newlines(s: str) -> str:
    """Turn literal backslash-n/tab sequences into real newlines/tabs and normalize CRLF."""
    s = s.replace("\r\n", "\n")
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", "\t")
    s = s.replace("\\r", "\r")
    return s

def layout_glyphs(text: str, font: ImageFont.FreeTypeFont, max_width: int, line_height: int, margin: int):
    """
    Layout text into (char, x, y) tuples with word wrapping.
    Returns (glyphs, total_height)
    """
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    tokens = []
    i = 0
    while i < len(text):
        if text[i] == "\n":
            tokens.append("\n")
            i += 1
        elif text[i].isspace():
            j = i
            while j < len(text) and text[j].isspace() and text[j] != "\n":
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            tokens.append(text[i:j])
            i = j

    lines: List[str] = []
    current = ""
    for tok in tokens:
        if tok == "\n":
            lines.append(current.rstrip())
            current = ""
            continue
        tentative = current + tok
        if tentative == "":
            continue
        w = draw.textlength(tentative, font=font)
        if w <= max_width or current == "":
            current = tentative
        else:
            lines.append(current.rstrip())
            current = tok.lstrip() if not tok.isspace() else ""

    if current:
        lines.append(current.rstrip())

    glyphs: List[Tuple[str, int, int]] = []
    y = margin
    for line in lines:
        x = margin
        for ch in line:
            w = draw.textlength(ch, font=font)
            glyphs.append((ch, int(x), int(y)))
            x += w
        y += line_height

    total_height = y - margin
    return glyphs, total_height

def _rgba_to_gif_p_with_transparency(im_rgba):
    """Convert an RGBA frame to paletted ('P') with a dedicated transparent index 255."""
    im_rgba = im_rgba.convert("RGBA")
    alpha = im_rgba.getchannel("A")
    im_rgb = im_rgba.convert("RGB")
    im_p = im_rgb.convert("P", palette=Image.ADAPTIVE, colors=255)  # Pillow may warn; acceptable.
    mask = alpha.point(lambda a: 255 if a == 0 else 0, mode="L")
    trans_index = 255
    im_p.paste(trans_index, box=None, mask=mask)
    im_p.info["transparency"] = trans_index
    return im_p

# why did ChatGPT do this? seems off to me.
# def generate_gif(
#     text: str,
#     out_path: str = "random_subset.gif",
#     width: int = 900,
#     height: int = 400,
#     font_path: str = "",
#     font_size: int = 48,
#     density: float = 0.33,
#     fps: int = 12,
#     duration_s: float = 4.0,
#     bg: str = "#0b0f19",
#     fg: str = "#e2e8f0",
#     margin: int = 24,
#     keep: int = 3,
#     line_height: int = 58,
#     seed: int = 12345,
#     skip_spaces: bool = True,
#     transparent: bool = False,
# ):


def generate_gif(
    text: str,
    out_path: str,
    width: int,
    height: int,
    font_path: str,
    font_size: int,
    density: float,
    fps: int,
    duration_s: float,
    bg: str,
    fg: str,
    margin,
    keep,
    line_height,
    seed,
    skip_spaces,
    transparent,
):
    """
    Produce an animated GIF where each frame shows a random subset of the letters.
    - density: 0..1 probability to draw each glyph per frame
    - fps, duration_s: control frame count and playback speed
    """
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
            print("font exception")

    glyphs, total_height = layout_glyphs(
        text=text, font=font, max_width=max(1, width - margin * 2),
        line_height=line_height, margin=margin
    )

    total_frames = max(1, int(round(fps * duration_s)))
    frame_ms = int(round(1000 / fps))
    frames: List[Image.Image] = []

    for i in range(total_frames):
        rng = random.Random(seed + i * 1013904223)
        if transparent:
            im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        else:
            im = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(im)
        
        det_threshold = margin + keep * draw.textlength('x', font=font)

        v_offset = 0
        if total_height < height - margin * 2:
            v_offset = int((height - total_height) / 2) - margin

        for ch, x, y in glyphs:
            if skip_spaces and ch.isspace():
                continue
            if x<det_threshold or rng.random() < density:
                draw.text((x, y + v_offset), ch, font=font, fill=fg)
        frames.append(im)

    if transparent:
        pal_frames = [_rgba_to_gif_p_with_transparency(f) for f in frames]
        pal_frames[0].save(
            out_path,
            save_all=True,
            append_images=pal_frames[1:],
            duration=frame_ms,
            loop=0,
            optimize=True,
            disposal=2,
            transparency=pal_frames[0].info.get("transparency", 255),
        )
    else:
        if len(frames) == 1:
            frames[0].save(out_path)
        else:
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_ms,
                loop=0,
                optimize=True,
                disposal=2,
            )
    return out_path

def positive_float(x: str) -> float:
    v = float(x)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Random-subset text -> animated GIF")
    p.add_argument("--text", type=str, default="Type something delightful here.\\nFlicker-fade letters dance!",
                   help="Text to animate. Use \\n for new lines (or pass --text_file).")
    p.add_argument("--text_file", type=str, default="", help="Optional path to a UTF-8 text file to read.")
    p.add_argument("--out", type=str, default="random_subset.gif", help="Output GIF path.")
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--font", type=str, default="", help="Path to a .ttf/.otf font file (defaults to DejaVuSans or PIL default).")
    p.add_argument("--font_size", type=int, default=48)
    p.add_argument("--line_height", type=int, default=58)
    p.add_argument("--density", type=float, default=0.33, help="Fraction of letters drawn per frame, 0..1")
    p.add_argument("--fps", type=int, default=12, help="Frames per second (also controls GIF frame duration).")
    p.add_argument("--seconds", type=positive_float, default=4.0, help="Animation duration in seconds.")
    p.add_argument("--bg", type=str, default="#000020", help="Background color (hex or CSS name).")
    p.add_argument("--fg", type=str, default="#ffffff", help="Text color (hex or CSS name).")
    p.add_argument("--margin", type=int, default=24, help="Padding around text block in px.")
    p.add_argument("--keep", type=int, default=3, help="How many characters per line to keep deterministically.")
    p.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility.")
    p.add_argument("--draw_spaces", action="store_true", help="If set, spaces will be drawn (invisible if same as bg).")
    p.add_argument("--transparent", action="store_true", help="Export with a transparent background (GIF 1-bit transparency).")
    return p

def main():
    parser = make_parser()
    args = parser.parse_args()

    text = _coerce_newlines(args.text)
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()

    path = generate_gif(
        text=text,
        out_path=args.out,
        width=args.width,
        height=args.height,
        font_path=args.font,
        font_size=args.font_size,
        density=args.density,
        fps=args.fps,
        duration_s=args.seconds,
        bg=args.bg,
        fg=args.fg,
        margin=args.margin,
        keep=args.keep,
        line_height=args.line_height,
        seed=args.seed,
        skip_spaces=not args.draw_spaces,
        transparent=args.transparent,
    )
    print(f"Wrote GIF to: {path}")

if __name__ == "__main__":
    main()
