"""Create a polished Arabic PDF explaining the NICE RAG pipeline from start to finish."""

from __future__ import annotations

from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "شرح_مشروع_NICE_RAG_من_البداية.pdf"

PAGE_W, PAGE_H = A4
MARGIN_X = 17 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

FONT_REGULAR = "TahomaArabic"
FONT_BOLD = "TahomaArabicBold"
FONT_MONO = "Courier"

NAVY = colors.HexColor("#0B2A4A")
BLUE = colors.HexColor("#176B9C")
TEAL = colors.HexColor("#10A6A0")
CYAN = colors.HexColor("#DFF6F5")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GOLD = colors.HexColor("#FFF5D8")
GOLD = colors.HexColor("#D9A62E")
GREEN = colors.HexColor("#198754")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#526170")
LINE = colors.HexColor("#D5E1E8")
WHITE = colors.white


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, r"C:\Windows\Fonts\tahoma.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, r"C:\Windows\Fonts\tahomabd.ttf"))


def visual(text: str) -> str:
    """Shape Arabic and apply bidi ordering for ReportLab's drawing engine."""
    return get_display(arabic_reshaper.reshape(text))


class ArabicParagraph(Flowable):
    def __init__(
        self,
        text: str,
        font_name: str = FONT_REGULAR,
        font_size: float = 10.3,
        leading: float = 17,
        color: colors.Color = INK,
        align: str = "right",
        left_padding: float = 0,
        right_padding: float = 0,
    ) -> None:
        super().__init__()
        self.text = " ".join(text.split())
        self.font_name = font_name
        self.font_size = font_size
        self.leading = leading
        self.color = color
        self.align = align
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.lines: list[str] = []

    def _wrap_lines(self, width: float) -> list[str]:
        usable = max(20, width - self.left_padding - self.right_padding)
        words = self.text.split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            trial = " ".join(current + [word])
            if current and pdfmetrics.stringWidth(visual(trial), self.font_name, self.font_size) > usable:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines or [""]

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        self.lines = self._wrap_lines(avail_width)
        self.height = len(self.lines) * self.leading
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFont(self.font_name, self.font_size)
        canvas.setFillColor(self.color)
        y = self.height - self.font_size
        for line in self.lines:
            shaped = visual(line)
            if self.align == "center":
                canvas.drawCentredString(self.width / 2, y, shaped)
            elif self.align == "left":
                canvas.drawString(self.left_padding, y, shaped)
            else:
                canvas.drawRightString(self.width - self.right_padding, y, shaped)
            y -= self.leading
        canvas.restoreState()


class StageBadge(Flowable):
    def __init__(self, number: str, label: str, color: colors.Color = TEAL) -> None:
        super().__init__()
        self.number = number
        self.label = label
        self.color = color
        self.height = 15 * mm

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        return avail_width, self.height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setFillColor(self.color)
        c.roundRect(0, 0, self.width, self.height, 4 * mm, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.circle(self.width - 9 * mm, self.height / 2, 5.2 * mm, fill=1, stroke=0)
        c.setFillColor(self.color)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(self.width - 9 * mm, self.height / 2 - 4, self.number)
        c.setFillColor(WHITE)
        c.setFont(FONT_BOLD, 13)
        c.drawRightString(self.width - 18 * mm, self.height / 2 - 5, visual(self.label))
        c.restoreState()


class PipelineDiagram(Flowable):
    STEPS = [
        ("1", "اختيار المصدر", "NICE NG151"),
        ("2", "تنزيل الملف", "PDF"),
        ("3", "تحويل وتنظيف", "Markdown + Pages"),
        ("4", "تقسيم منظم", "178 Chunks"),
        ("5", "نطاق النسخة الأولى", "33 Chunks"),
        ("6", "تحويل المعنى لأرقام", "Embeddings - 384"),
        ("7", "تخزين وبحث", "Chroma"),
        ("8", "اختبار الاسترجاع", "5 / 5 Passed"),
        ("9", "الخطوة القادمة", "RAG Answer"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.height = 158 * mm

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        return avail_width, self.height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        box_h = 13.5 * mm
        gap = 3.3 * mm
        box_w = self.width - 24 * mm
        x = 12 * mm
        y = self.height - box_h
        for index, (number, ar_label, en_label) in enumerate(self.STEPS):
            fill = PALE_BLUE if index % 2 == 0 else CYAN
            border = BLUE if index < 5 else TEAL
            c.setFillColor(fill)
            c.setStrokeColor(border)
            c.setLineWidth(1)
            c.roundRect(x, y, box_w, box_h, 3 * mm, fill=1, stroke=1)
            c.setFillColor(border)
            c.circle(x + box_w - 8 * mm, y + box_h / 2, 4.7 * mm, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 9)
            c.drawCentredString(x + box_w - 8 * mm, y + box_h / 2 - 3.2, number)
            c.setFillColor(NAVY)
            c.setFont(FONT_BOLD, 10.5)
            c.drawRightString(x + box_w - 16 * mm, y + box_h / 2 + 1, visual(ar_label))
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 8.2)
            c.drawString(x + 5 * mm, y + box_h / 2 - 2.4, en_label)
            if index < len(self.STEPS) - 1:
                c.setStrokeColor(LINE)
                c.setLineWidth(1.4)
                mid = x + box_w / 2
                c.line(mid, y, mid, y - gap + 1.2 * mm)
                c.line(mid, y - gap + 1.2 * mm, mid - 1.5 * mm, y - gap + 3 * mm)
                c.line(mid, y - gap + 1.2 * mm, mid + 1.5 * mm, y - gap + 3 * mm)
            y -= box_h + gap
        c.restoreState()


def rtl(text: str, **kwargs) -> ArabicParagraph:
    return ArabicParagraph(text, **kwargs)


def section(number: str, title: str, color: colors.Color = TEAL) -> list[Flowable]:
    return [StageBadge(number, title, color), Spacer(1, 4 * mm)]


def body(text: str) -> list[Flowable]:
    return [rtl(text), Spacer(1, 2.2 * mm)]


def bullets(items: list[str]) -> list[Flowable]:
    flowables: list[Flowable] = []
    for item in items:
        flowables.append(rtl("• " + item, font_size=9.8, leading=16, right_padding=2 * mm))
        flowables.append(Spacer(1, 0.8 * mm))
    flowables.append(Spacer(1, 1.2 * mm))
    return flowables


def callout(title: str, text: str, tone: str = "blue") -> Table:
    palette = {
        "blue": (PALE_BLUE, BLUE),
        "teal": (CYAN, TEAL),
        "gold": (PALE_GOLD, GOLD),
    }
    bg, accent = palette[tone]
    table = Table(
        [
            [rtl(title, font_name=FONT_BOLD, font_size=10.5, leading=15, color=accent)],
            [rtl(text, font_size=9.5, leading=15.5, color=INK)],
        ],
        colWidths=[CONTENT_W],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.8, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 7 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7 * mm),
                ("TOPPADDING", (0, 0), (-1, 0), 4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 1.5 * mm),
                ("TOPPADDING", (0, 1), (-1, 1), 1 * mm),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 4 * mm),
            ]
        )
    )
    return table


def code_block(lines: list[str]) -> Table:
    style = ParagraphStyle(
        "Code",
        fontName=FONT_MONO,
        fontSize=8.6,
        leading=13,
        textColor=colors.HexColor("#DCEAF3"),
        alignment=TA_LEFT,
    )
    content = Paragraph("<br/>".join(line.replace("&", "&amp;") for line in lines), style)
    table = Table([[content]], colWidths=[CONTENT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("BOX", (0, 0), (-1, -1), 0, NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 6 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
            ]
        )
    )
    return table


def status_table() -> Table:
    rows = [
        ["Status", "Output", visual("المرحلة")],
        ["DONE", "Official NICE NG151 PDF", visual("اختيار وتنزيل المصدر")],
        ["DONE", "Markdown + pages JSONL", visual("تحويل الملف")],
        ["DONE", "178 total / 33 MVP", visual("بناء الـ Chunks")],
        ["DONE", "384-d embeddings", visual("إنشاء الـ Embeddings")],
        ["DONE", "Local Chroma DB", visual("التخزين والبحث")],
        ["DONE", "5 / 5 retrieval tests", visual("الاختبار")],
        ["NEXT", "LLM answer + citations", visual("توليد إجابة RAG")],
    ]
    table = Table(rows, colWidths=[25 * mm, 72 * mm, CONTENT_W - 97 * mm], repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (1, -1), "Helvetica"),
        ("FONTNAME", (2, 1), (2, -1), FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2 * mm),
    ]
    for row in range(1, len(rows)):
        commands.append(("BACKGROUND", (0, row), (-1, row), PALE_BLUE if row % 2 else colors.white))
        commands.append(("TEXTCOLOR", (0, row), (0, row), GREEN if rows[row][0] == "DONE" else GOLD))
        commands.append(("FONTNAME", (0, row), (0, row), "Helvetica-Bold"))
    table.setStyle(TableStyle(commands))
    return table


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 13 * mm, PAGE_W, 13 * mm, fill=1, stroke=0)
        canvas.setFont(FONT_BOLD, 8.5)
        canvas.setFillColor(WHITE)
        canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 8.5 * mm, visual("شرح مشروع NICE RAG"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN_X, PAGE_H - 8.5 * mm, "Colorectal Cancer - NG151")
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN_X, 13 * mm, PAGE_W - MARGIN_X, 13 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(PAGE_W / 2, 8.5 * mm, str(doc.page))
    canvas.restoreState()


def build_story() -> list[Flowable]:
    story: list[Flowable] = []

    # Cover
    story += [Spacer(1, 26 * mm)]
    story.append(rtl("شرح مشروع NICE RAG من البداية", font_name=FONT_BOLD, font_size=25, leading=35, color=NAVY, align="center"))
    story.append(Spacer(1, 5 * mm))
    story.append(rtl("من اختيار Guideline سرطان القولون والمستقيم إلى البحث الدلالي وتجهيز نظام الإجابة بالمصادر", font_size=13, leading=23, color=BLUE, align="center", left_padding=8 * mm, right_padding=8 * mm))
    story.append(Spacer(1, 15 * mm))
    cover_box = Table(
        [
            [Paragraph("NICE NG151", ParagraphStyle("covercode", fontName="Helvetica-Bold", fontSize=18, textColor=WHITE, alignment=TA_CENTER))],
            [rtl("Colorectal Cancer Guideline", font_name=FONT_BOLD, font_size=13, leading=19, color=WHITE, align="center")],
        ],
        colWidths=[115 * mm],
    )
    cover_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("BOX", (0, 0), (-1, -1), 1.5, TEAL),
        ("TOPPADDING", (0, 0), (-1, -1), 6 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6 * mm),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(Table([[cover_box]], colWidths=[CONTENT_W], style=[("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(Spacer(1, 19 * mm))
    story.append(rtl("هذا الملف يشرح ماذا فعلنا، لماذا فعلناه، ما الملفات الناتجة، وكيف تتحول أسئلة المستخدم إلى نتائج موثقة من الـ Guideline.", font_size=11.5, leading=20, color=MUTED, align="center", left_padding=13 * mm, right_padding=13 * mm))
    story.append(Spacer(1, 26 * mm))
    story.append(rtl("نسخة تعليمية مبسطة - مناسبة للمراجعة والعرض في الهاكاثون", font_size=9.5, leading=15, color=TEAL, align="center"))
    story.append(PageBreak())

    # Overview
    story += section("0", "الصورة الكبيرة", NAVY)
    story += body("هدف المشروع هو بناء نظام طبي يجيب عن الأسئلة باستخدام Guideline موثوق، بدل الاعتماد على ذاكرة النموذج أو مقالات عشوائية. كل إجابة في النهاية يجب أن يمكن تتبعها إلى التوصية والصفحة والمصدر.")
    story.append(callout("الفكرة في سطر واحد", "نحوّل الـ PDF إلى أجزاء صغيرة منظمة، نحوّل معنى كل جزء إلى Vector، نخزنه، ثم نبحث عن أقرب أجزاء لسؤال المستخدم ونستخدمها في تكوين الإجابة.", "teal"))
    story.append(Spacer(1, 5 * mm))
    story.append(PipelineDiagram())
    story.append(PageBreak())

    # Source selection
    story += section("1", "اختيار فكرة السرطان والـ Guideline")
    story += body("اخترنا سرطان القولون والمستقيم - Colorectal Cancer - واستخدمنا NICE NG151 كمرجع أساسي. NICE مصدر رسمي، والتوصيات فيه مرقمة ومنظمة إلى Sections وSubsections، لذلك يناسب الاستخراج والبحث وإظهار الـ Citation.")
    story += bullets([
        "مصدر طبي رسمي وموثوق بدل صفحات إنترنت غير معروفة.",
        "التوصيات مرقمة مثل 1.3.1 و 1.6.1.",
        "الصفحات والجداول واضحة ويمكن الرجوع إليها.",
        "يمكن إظهار رقم التوصية ورقم الصفحة في الإجابة النهائية.",
    ])
    story.append(callout("ليه العمق أفضل من كثرة المصادر؟", "في نسخة الهاكاثون الأولى، توثيق Guideline واحد بشكل جيد أفضل من جمع مصادر كثيرة بجودة غير مضمونة. بعد نجاح الـ MVP يمكن إضافة Guidelines أخرى.", "gold"))
    story.append(Spacer(1, 6 * mm))

    story += section("2", "تنزيل الـ PDF وتسجيل المصدر", BLUE)
    story += body("نزّلنا ملف الـ PDF الرسمي وحفظناه داخل data/raw. كلمة raw تعني النسخة الأصلية كما جاءت من المصدر من غير تعديل، حتى نرجع إليها عند مراجعة أي نص أو جدول أو رقم صفحة.")
    story.append(code_block([
        "data/raw/nice_ng151_colorectal_cancer.pdf",
        "data/source_manifest.json",
    ]))
    story.append(Spacer(1, 4 * mm))
    story += body("أنشأنا Source Manifest لتسجيل اسم الوثيقة، كود NG151، المرض، رابط المصدر واسم الملف. هذا يحقق traceability: أي معلومة يمكن تتبعها إلى الوثيقة الأصلية.")
    story.append(PageBreak())

    # Parsing
    story += section("3", "تحويل الـ PDF إلى Markdown")
    story += body("الـ PDF ممتاز للقراءة البشرية لكنه صعب في المعالجة البرمجية. لذلك استخدمنا سكريبت يحوّل المحتوى إلى Markdown منظم، بحيث تظهر العناوين والتوصيات والجداول كنص يمكن فحصه وتقسيمه.")
    story.append(code_block([
        "scripts/01_pdf_to_markdown.py",
        "data/processed/nice_ng151_colorectal_cancer.md",
    ]))
    story.append(Spacer(1, 4 * mm))
    story.append(callout("ليه Markdown؟", "لأنه نص بسيط ومنظم، يحافظ على العناوين والقوائم والجداول، ويسهل البحث والمراجعة وبناء الـ Chunks.", "blue"))
    story.append(Spacer(1, 6 * mm))

    story += section("4", "حفظ الصفحات وفحص الجودة", BLUE)
    story += body("حفظنا كل صفحة كسجل مستقل في ملف JSONL. وجود page_number مع النص ضروري حتى تخرج الإجابة لاحقًا ومعها رقم الصفحة الصحيح.")
    story.append(code_block(["data/processed/nice_ng151_colorectal_cancer_pages.jsonl"]))
    story.append(Spacer(1, 4 * mm))
    story += body("بعد التحويل شغّلنا Quality Assurance للتأكد من أن الصفحات ليست فارغة، وأن التوصيات المرقمة والعناوين والجداول لم تختف أثناء الاستخراج.")
    story.append(callout("قاعدة مهمة", "إذا كان النص المستخرج خطأ، فكل ما بعده سيكون خطأ: Chunks خطأ، Embeddings خطأ، ثم إجابات خطأ. لذلك فحص التحويل خطوة أساسية وليست اختيارية.", "gold"))
    story.append(PageBreak())

    # Chunking
    story += section("5", "تقسيم النص إلى Chunks")
    story += body("الـ Chunk هو جزء صغير مستقل من الـ Guideline. بدل إرسال الملف كله عند كل سؤال، نبحث عن الأجزاء المرتبطة بالسؤال فقط.")
    story.append(code_block([
        "scripts/02_build_chunks.py",
        "data/chunks/nice_ng151_colorectal_cancer_chunks.jsonl",
    ]))
    story.append(Spacer(1, 4 * mm))
    story += bullets([
        "كل Recommendation بقي Chunk مستقل حتى لا يتقطع المعنى.",
        "العنوان ورقم الصفحة ورقم التوصية محفوظون مع النص.",
        "الجداول مقسمة حسب الصفحة وأجزاؤها مرتبطة ببعض.",
        "جدول علاج سرطان المستقيم المبكر مرتبط بالتوصية 1.3.3.",
    ])
    story.append(callout("لماذا لم نقسم كل 500 كلمة؟", "التقسيم العشوائي قد يقطع توصية طبية من المنتصف. استخدمنا Structure-aware Chunking يحترم بنية NICE ويحافظ على الوحدة الدلالية لكل توصية.", "teal"))
    story.append(Spacer(1, 6 * mm))

    story += section("6", "178 Chunk في المصدر و 33 للـ MVP", BLUE)
    story += body("نتج من الوثيقة 178 Chunk تشمل توصيات وجداول ونصوص داعمة. للنسخة الأولى اخترنا 33 Chunk فقط بعلامة in_initial_scope=true.")
    story += bullets([
        "الـ 33 تشمل الأقسام المتفق عليها والتوصيات المهمة والجداول المرتبطة بها.",
        "النطاق الصغير يسمح باختبار دقيق وإصلاح الأخطاء بسرعة.",
        "بعد نجاح الـ MVP يمكن توسيع الفهرس إلى كل التوصيات أو إلى الـ 178 Chunk.",
    ])
    story.append(PageBreak())

    # Embeddings
    story += section("7", "تجهيز النص المناسب للـ Embedding")
    story += body("كل Chunk له content وهو النص الأصلي، وله text وهو النص المجهز للبحث. حقل text يضيف اسم الوثيقة والقسم والعنوان والصفحات قبل المحتوى، حتى يفهم الموديل سياق التوصية بشكل أفضل.")
    story.append(code_block([
        "Document: Colorectal cancer (NG151)",
        "Pages: 26-26",
        "Section: 1.6 Ongoing care and support",
        "Recommendation: 1.6.1",
        "For people who have had potentially curative surgical treatment...",
    ]))
    story.append(Spacer(1, 6 * mm))

    story += section("8", "ما هو الـ Embedding؟", TEAL)
    story += body("الـ Embedding هو تحويل معنى النص إلى قائمة أرقام. كل Chunk عندنا يتحول إلى Vector طوله 384 رقم. النصوص المتقاربة في المعنى تنتج Vectors متقاربة، حتى لو كان السؤال بالعربي والنص بالإنجليزي.")
    story.append(code_block(["[0.021, -0.047, 0.012, 0.063, ...]  -> 384 values"]))
    story.append(Spacer(1, 4 * mm))
    story.append(callout("مهم: المكتبة ليست الموديل", "sentence-transformers هي المكتبة التي نشغّل بها النماذج. الموديل الذي اخترناه داخلها هو intfloat/multilingual-e5-small.", "blue"))
    story.append(PageBreak())

    # Model
    story += section("9", "اختيار موديل Multilingual E5")
    story += body("اخترنا intfloat/multilingual-e5-small لأنه متعدد اللغات، يفهم العربي والإنجليزي، ومتدرب خصيصًا على Retrieval من سؤال إلى فقرة. يعمل محليًا ولا يحتاج API key لخدمة Embeddings مدفوعة.")
    story += bullets([
        "مناسب لأن الـ Guideline إنجليزي وأسئلة الديمو قد تكون عربية.",
        "متخصص في query-to-passage retrieval بدل تشابه الجمل فقط.",
        "يعطي Vectors بطول 384، وهو مناسب لحجم مشروع الهاكاثون.",
        "يعمل Local وتتم إعادة استخدام الموديل من الـ cache بعد أول تنزيل.",
    ])
    story.append(callout("لماذا غيّرنا الموديل الأول؟", "الموديل الأول العام نجح في 4 من 5 اختبارات، لكنه لم يرجع توصية المتابعة الصحيحة لسؤال عربي ضمن أول 5 نتائج. E5 أعاد التوصية الصحيحة في المركز الأول، فأصبح الاختبار 5 من 5.", "gold"))
    story.append(Spacer(1, 6 * mm))

    story += section("10", "الفرق بين query و passage", BLUE)
    story += body("تعليمات E5 الرسمية تطلب إضافة prefix داخلي. نستخدم passage قبل النص المخزن و query قبل سؤال المستخدم. هذه الكلمات لا تظهر في واجهة المستخدم، لكنها تساعد الموديل يفهم وظيفة كل نص.")
    story.append(code_block([
        "passage: Offer follow-up for detection of local recurrence...",
        "query: What follow-up is recommended after curative surgery?",
    ]))
    story.append(PageBreak())

    # Chroma
    story += section("11", "تخزين الـ Vectors في Chroma")
    story += body("استخدمنا Chroma كـ Vector Database محلية. السكريبت يحسب Embeddings للـ 33 Chunk ثم يخزن النص والـ Vector والـ Metadata في Collection اسمها nice_ng151_colorectal.")
    story.append(code_block([
        "scripts/03_build_vector_index.py",
        "data/vector_store/chroma/",
        "Collection: nice_ng151_colorectal",
        "Vectors: 33 | Dimension: 384 | Metric: cosine",
    ]))
    story.append(Spacer(1, 5 * mm))
    story.append(callout("ليه Vector Database؟", "البحث بالكلمات قد لا يطابق السؤال العربي مع النص الإنجليزي. البحث بالـ Vector يقارن المعنى، لذلك يفهم أن المتابعة قريبة من Follow-up وأن الجراحة قريبة من Surgical treatment.", "teal"))
    story.append(Spacer(1, 6 * mm))

    story += section("12", "الـ Metadata والـ Citations", BLUE)
    story += body("الـ Vector يجد النص المناسب، لكن الـ Metadata هي التي تخبرنا من أي وثيقة وقسم وصفحة وتوصية جاء النص. خزّنا الحقول المطلوبة في العرض وأضفنا حقولًا أخرى مفيدة.")
    story += bullets([
        "document_name - اسم الوثيقة.",
        "section_title - عنوان القسم.",
        "page_number و page_start و page_end - الصفحات.",
        "chunk_id و recommendation_ids - هوية الجزء والتوصية.",
        "source_url - رابط NICE الرسمي.",
        "content_type و linked_chunk_ids - نوع المحتوى وروابط الجداول.",
    ])
    story.append(PageBreak())

    # Retrieval
    story += section("13", "كيف يحدث البحث الدلالي؟")
    story += body("عند وصول سؤال المستخدم، يتحول السؤال إلى Vector بالموديل نفسه. تقارن Chroma هذا الـ Vector بالـ Vectors المخزنة باستخدام cosine similarity، ثم تعيد أفضل النتائج مرتبة من الأقرب للأبعد.")
    story += bullets([
        "السؤال يحصل على prefix باسم query.",
        "الموديل ينتج Vector طوله 384.",
        "Chroma تقارن السؤال بالـ 33 Vector.",
        "تعود أفضل 3 إلى 5 Chunks مع النص والـ Metadata.",
        "نستخدم الصفحة ورقم التوصية لعرض المصدر.",
    ])
    story.append(code_block([
        "Arabic query: follow-up after colorectal cancer surgery",
        "Top result: ng151-rec-1-6-1",
        "Recommendation: 1.6.1 | Page: 26",
    ]))
    story.append(Spacer(1, 6 * mm))

    story += section("14", "اختبار الاسترجاع", TEAL)
    story += body("جهزنا خمسة أسئلة بالإنجليزي والعربي. لا يكفي أن يعمل الكود؛ اعتبرنا الاختبار ناجحًا فقط إذا ظهر Chunk متوقع وصحيح ضمن أول خمس نتائج.")
    story += bullets([
        "علاج سرطان المستقيم المبكر بالإنجليزي.",
        "الانسداد الحاد في الجانب الأيسر من الأمعاء الغليظة بالإنجليزي.",
        "المتابعة بعد الجراحة العلاجية بالإنجليزي.",
        "علاج سرطان المستقيم المبكر بالعربي.",
        "المتابعة بعد الجراحة العلاجية بالعربي.",
    ])
    story.append(callout("النتيجة النهائية", "5 / 5 اختبارات نجحت. سؤال المتابعة بالعربي والإنجليزي أعاد Recommendation 1.6.1 في المركز الأول.", "teal"))
    story.append(PageBreak())

    # Status and next step
    story += section("15", "ما الذي انتهى وما الخطوة التالية؟", NAVY)
    story.append(status_table())
    story.append(Spacer(1, 7 * mm))
    story += body("إحنا خلصنا إعداد المصدر، التحويل، الـ Chunking، الـ Embeddings، التخزين، والبحث. الخطوة التالية هي RAG Answer Generation: نأخذ أفضل Chunks ونرسلها مع السؤال إلى LLM ليكتب إجابة مقيدة بالمصدر.")
    story.append(callout("شكل الإجابة النهائية المتوقع", "إجابة واضحة للمستخدم، ثم Recommendation ID ورقم الصفحة ورابط NICE. النموذج لا يفترض معلومات من ذاكرته؛ يستخدم الـ Context المسترجع فقط.", "gold"))
    story.append(Spacer(1, 7 * mm))
    story.append(code_block([
        "User question",
        "  -> Embed query",
        "  -> Retrieve top chunks from Chroma",
        "  -> Send question + context to LLM",
        "  -> Answer + recommendation + page + source",
    ]))
    story.append(PageBreak())

    # Files and commands
    story += section("16", "دليل الملفات والأوامر", BLUE)
    story += body("الملفات التالية تمثل كل مرحلة من المشروع، ويمكن استخدامها في العرض أو عند إعادة تشغيل الـ Pipeline.")
    story.append(code_block([
        "data/raw/nice_ng151_colorectal_cancer.pdf",
        "data/source_manifest.json",
        "data/processed/nice_ng151_colorectal_cancer.md",
        "data/processed/nice_ng151_colorectal_cancer_pages.jsonl",
        "data/chunks/nice_ng151_colorectal_cancer_chunks.jsonl",
        "data/vector_store/chroma/",
        "data/vector_store/index_report.json",
        "data/evaluation/retrieval_report.json",
    ]))
    story.append(Spacer(1, 6 * mm))
    story += body("أوامر التشغيل الأساسية:")
    story.append(code_block([
        ".\\.venv\\Scripts\\python.exe scripts\\01_pdf_to_markdown.py",
        ".\\.venv\\Scripts\\python.exe scripts\\02_build_chunks.py",
        ".\\.venv\\Scripts\\python.exe scripts\\03_build_vector_index.py",
        ".\\.venv\\Scripts\\python.exe scripts\\04_test_retrieval.py",
    ]))
    story.append(Spacer(1, 6 * mm))
    story += body("لتجربة سؤال يدوي:")
    story.append(code_block([
        ".\\.venv\\Scripts\\python.exe scripts\\04_test_retrieval.py --query",
        '"What follow-up is recommended after curative colorectal cancer surgery?"',
    ]))
    story.append(Spacer(1, 8 * mm))
    story.append(callout("الخلاصة", "المشروع الآن يمتلك Knowledge Base طبية صغيرة لكنها منظمة ومختبرة. الخطوة التالية ليست البحث، لأن البحث انتهى ونجح؛ الخطوة التالية هي بناء Prompt وربط نتائج Chroma بالـ LLM لإنتاج إجابة موثقة.", "teal"))

    return story


def main() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="NICE RAG Project Explanation - Arabic",
        author="Creativa Hackathon Team",
        subject="End-to-end explanation of the colorectal cancer guideline RAG pipeline",
    )
    doc.build(build_story(), onFirstPage=header_footer, onLaterPages=header_footer)
    print("PDF created successfully in output/pdf")


if __name__ == "__main__":
    main()
