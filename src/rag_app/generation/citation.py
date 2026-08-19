"""Citation formatting for bilingual document references."""

from __future__ import annotations

from .. import config


ARABIC_SECTIONS = {
    "1.1 Reduction in risk of colorectal cancer in people with Lynch syndrome": "1.1 تقليل خطر سرطان القولون والمستقيم لدى الأشخاص المصابين بمتلازمة لينش",
    "1.2 Information for people with colorectal cancer": "1.2 معلومات للأشخاص المصابين بسرطان القولون والمستقيم",
    "1.3 Management of local disease": "1.3 علاج المرض الموضعي",
    "1.3 Colorectal cancer recognition and referral": "1.3 التعرّف على سرطان القولون والمستقيم والإحالة",
    "1.4 Molecular biomarkers to guide systemic anticancer therapy": "1.4 المؤشرات الحيوية الجزيئية لتوجيه العلاج الجهازي المضاد للسرطان",
    "1.5 Management of advanced or metastatic colorectal cancer": "1.5 علاج سرطان القولون والمستقيم المتقدم أو المنتشر",
    "Follow-up for detection of local recurrence and distant metastases": "المتابعة لاكتشاف الانتكاس الموضعي والنقائل البعيدة",
    "People with rectal cancer": "الأشخاص المصابون بسرطان المستقيم",
    "People with colon cancer": "الأشخاص المصابون بسرطان القولون",
    "People with locally advanced or recurrent rectal cancer": "الأشخاص المصابون بسرطان المستقيم المتقدم موضعياً أو الناكس",
    "Surgery for people with rectal cancer": "الجراحة للأشخاص المصابين بسرطان المستقيم",
    "Surgical technique for people with rectal cancer": "التقنية الجراحية لسرطان المستقيم",
    "Surgical technique for people with colon cancer": "التقنية الجراحية لسرطان القولون",
    "Preoperative treatment for people with rectal cancer": "العلاج قبل الجراحة لسرطان المستقيم",
    "Other systemic anticancer therapy for untreated disease": "العلاج الجهازي المضاد للسرطان الآخر للمرض غير المعالج",
    "BRAF V600E mutation-positive disease": "المرض الإيجابي لطفرة BRAF V600E",
    "Neurotrophic tyrosine receptor kinase (NTRK) fusion-positive solid tumours": "الأورام الصلبة الإيجابية لاندماج NTRK",
    "People with metastatic colorectal cancer in the lung": "سرطان القولون والمستقيم المنتشر إلى الرئة",
    "People with metastatic colorectal cancer in the peritoneum": "سرطان القولون والمستقيم المنتشر إلى الصفاق",
}


def citation_for(row: dict, arabic: bool) -> str:
    """Create the required document / section / page citation."""
    if arabic:
        section = ARABIC_SECTIONS.get(row["section_title"], row["section_title"])
        if row["document_name"] == config.NG12_DOCUMENT_NAME:
            return f"[NICE NG12: الاشتباه بالسرطان والتعرّف عليه والإحالة، القسم: {section}، الصفحة: {row['page_number']}]"
        return f"[NICE NG151: سرطان القولون والمستقيم، القسم: {section}، الصفحة: {row['page_number']}]"
    return (
        f"[{row['document_name']}, Section: {row['section_title']}, "
        f"Page: {row['page_number']}]"
    )
