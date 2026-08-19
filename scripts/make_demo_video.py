"""Create a short silent MP4 demo for the colorectal cancer RAG assistant."""

from __future__ import annotations

from pathlib import Path
import textwrap

import arabic_reshaper
import imageio.v2 as imageio
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "demo"
VIDEO_PATH = OUT_DIR / "colorectal_rag_demo.mp4"

W, H = 1280, 720
FPS = 24

FONT = Path("C:/Windows/Fonts/tahoma.ttf")
BOLD = Path("C:/Windows/Fonts/tahomabd.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def rtl(text: str) -> str:
    if any("\u0600" <= char <= "\u06ff" for char in text):
        return get_display(arabic_reshaper.reshape(text))
    return text


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    fill: str,
    size: int = 26,
    bold: bool = False,
    align: str = "right",
    line_gap: int = 10,
) -> int:
    current_font = font(size, bold)
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textlength(rtl(test), font=current_font) <= width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    x, y = xy
    for line in lines:
        shaped = rtl(line)
        line_width = draw.textlength(shaped, font=current_font)
        draw_x = x + width - line_width if align == "right" else x
        draw.text((draw_x, y), shaped, font=current_font, fill=fill)
        y += size + line_gap
    return y


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=None, radius=24, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def background() -> Image.Image:
    img = Image.new("RGB", (W, H), "#07111f")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 72), fill="#081a2d")
    draw.text((38, 22), "Colorectal Cancer Assistant", font=font(24, True), fill="#e7f0fb")
    draw.text((980, 25), "NICE-grounded RAG", font=font(18), fill="#8dc8ff")
    draw.line((0, 72, W, 72), fill="#203b58", width=2)
    return img


def title_frame(title: str, subtitle: str) -> Image.Image:
    img = background()
    draw = ImageDraw.Draw(img)
    draw.text((64, 150), title, font=font(46, True), fill="#e7f0fb")
    draw.text((66, 220), subtitle, font=font(25), fill="#8ca6c1")
    rounded(draw, (64, 315, 1216, 500), "#0d1b2d", "#203b58")
    steps = [
        "PDF parsing + cleaning",
        "Structure-aware recursive chunks",
        "multilingual-e5-base embeddings",
        "Chroma vector database",
        "Qwen LLM with citations or safe refusal",
    ]
    x = 100
    for step in steps:
        draw.text((x, 370), step, font=font(19, True), fill="#e7f0fb")
        x += 215
    draw.text((64, 620), "Demo target: under 3 minutes", font=font(22, True), fill="#46d6a0")
    return img


def chat_frame(question: str, answer: str, source: str | None, badge: str) -> Image.Image:
    img = background()
    draw = ImageDraw.Draw(img)
    draw.text((62, 102), badge, font=font(22, True), fill="#46d6a0")

    # User bubble
    bubble_w = 660
    rounded(draw, (W - bubble_w - 70, 150, W - 70, 225), "#1e5a91", None, 22)
    draw_wrapped(draw, question, (W - bubble_w - 45, 172), bubble_w - 55, "#ffffff", 22)

    # Assistant bubble
    rounded(draw, (70, 275, 1120, 560), "#0a2826", "#2b805f", 24)
    draw.text((1020, 300), rtl("الإجابة"), font=font(20, True), fill="#71e5b9")
    draw_wrapped(draw, answer, (115, 345), 940, "#e7f0fb", 23)

    if source:
        rounded(draw, (765, 475, 1060, 526), "#111b28", "#3b536c", 20)
        draw.text((935, 489), "NICE source", font=font(17, True), fill="#d6e3f1")
        draw.text((795, 489), source, font=font(17), fill="#8dc8ff")
    else:
        draw.text((115, 598), rtl("لا يوجد مصدر لأن السؤال خارج النص المفهرس."), font=font(18), fill="#8ca6c1")

    draw.line((70, 585, 1120, 585), fill="#1d5045", width=1)
    draw.text(
        (545, 610),
        rtl("تنبيه: الإجابة للمعلومات فقط ولا تغني عن استشارة طبيب."),
        font=font(17),
        fill="#9db8b0",
    )
    return img


def analysis_frame() -> Image.Image:
    img = background()
    draw = ImageDraw.Draw(img)
    draw.text((60, 105), "Evaluation snapshot", font=font(34, True), fill="#e7f0fb")
    metrics = [
        ("Found Rate@5", "98.36%"),
        ("Mean Precision@5", "25.08%"),
        ("MAP@5", "78.92%"),
        ("MRR", "82.05%"),
    ]
    x = 60
    for label, value in metrics:
        rounded(draw, (x, 180, x + 270, 330), "#0d1b2d", "#203b58", 18)
        draw.text((x + 25, 210), label, font=font(19), fill="#8ca6c1")
        draw.text((x + 25, 252), value, font=font(34, True), fill="#e7f0fb")
        x += 300

    rounded(draw, (60, 395, 1220, 585), "#0d1b2d", "#203b58", 18)
    lines = [
        "Test set: 130 questions | 122 in-scope | 8 out-of-scope",
        "Arabic found rate: 61/61 | English found rate: 59/61",
        "Production retrieval stays fixed at Top-k = 5",
        "Out-of-scope and unsafe medicine questions are refused safely.",
    ]
    y = 425
    for line in lines:
        draw.text((95, y), line, font=font(24), fill="#e7f0fb")
        y += 38
    return img


def outro_frame() -> Image.Image:
    img = background()
    draw = ImageDraw.Draw(img)
    draw.text((70, 160), "Key takeaway", font=font(44, True), fill="#e7f0fb")
    draw_wrapped(
        draw,
        "النظام لا يجاوب من ذاكرته العامة. إذا وجد دليلًا مناسبًا داخل NICE يعرض الإجابة مع المصدر، وإذا كان السؤال خارج النطاق أو يحتاج نصيحة طبية شخصية يرفض بأمان.",
        (85, 250),
        1100,
        "#e7f0fb",
        30,
    )
    draw.text((70, 580), "Grounded answers. Clear citations. Safe refusal.", font=font(30, True), fill="#46d6a0")
    return img


def fade_frames(a: Image.Image, b: Image.Image, frames: int = 10):
    for i in range(frames):
        yield Image.blend(a, b, i / frames)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = [
        (
            title_frame(
                "Colorectal Cancer RAG Assistant",
                "A bilingual clinical guideline assistant grounded only in NICE evidence.",
            ),
            8,
        ),
        (
            chat_frame(
                "ما هي المتابعة المطلوبة بعد الجراحة العلاجية لسرطان القولون والمستقيم؟",
                "توصي NICE بالمتابعة لمدة 3 سنوات لاكتشاف الانتكاس الموضعي أو النقائل البعيدة، وتشمل CEA وتصوير CT للصدر والبطن والحوض.",
                "NG151 · 1.6.1 · Page 26",
                "Best case: in-scope question with citation",
            ),
            14,
        ),
        (
            chat_frame(
                "ممكن العب رياضه؟",
                "دليل NICE لا يعطي برنامجًا رياضيًا محددًا، لكنه يوصي أن يعطي فريق الرعاية نصائح عن تعديل النشاط البدني بعد العلاج للحفاظ على جودة الحياة.",
                "NG151 · 1.2.7 · Page 9",
                "Arabic general question: mapped to guideline evidence",
            ),
            12,
        ),
        (
            chat_frame(
                "اعمل اي لو عندي سرطان كبد؟",
                "هذا السؤال خارج نطاق النظام الحالي؛ قاعدة المعرفة المفهرسة خاصة بسرطان القولون والمستقيم فقط، لذلك لا أقدم إجابة عن نوع سرطان آخر من هذه الداتا.",
                None,
                "Worst case: out-of-scope disease",
            ),
            12,
        ),
        (
            chat_frame(
                "لو حاسس بألم ممكن اخد مسكن؟",
                "الدليل المفهرس لا يحتوي على توصية محددة عن اختيار مسكن أو جرعته. لا تبدأ أو تغيّر أي دواء بدون سؤال الطبيب أو الصيدلي.",
                None,
                "Safety case: no medicine or dosage advice",
            ),
            12,
        ),
        (analysis_frame(), 12),
        (outro_frame(), 8),
    ]

    with imageio.get_writer(str(VIDEO_PATH), fps=FPS, codec="libx264", quality=8) as writer:
        previous = None
        for image, seconds in scenes:
            if previous is not None:
                for frame in fade_frames(previous, image, 8):
                    writer.append_data(np.asarray(frame))
            for _ in range(seconds * FPS):
                writer.append_data(np.asarray(image))
            previous = image

    print(VIDEO_PATH)


if __name__ == "__main__":
    main()
