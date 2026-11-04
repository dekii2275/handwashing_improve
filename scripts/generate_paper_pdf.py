import os
import sys
import re
import base64
import subprocess
import markdown

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(ROOT, "paper_assets")
FIGURES_DIR = os.path.join(ASSETS_DIR, "figures")
CHROME_PATH = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

def img_to_base64(img_path):
    if not os.path.exists(img_path):
        print(f"Warning: image {img_path} not found!")
        return ""
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def render_math_safe_markdown(md_text):
    math_blocks = []
    def save_block(match):
        math_blocks.append(match.group(1))
        return f"%%MATHBLOCK_{len(math_blocks)-1}%%"
    md_text = re.sub(r'\$\$(.*?)\$\$', save_block, md_text, flags=re.DOTALL)

    math_inlines = []
    def save_inline(match):
        math_inlines.append(match.group(1))
        return f"%%MATHINLINE_{len(math_inlines)-1}%%"
    md_text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', save_inline, md_text)

    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    for i, b in enumerate(math_blocks):
        clean_b = b.strip()
        html_body = html_body.replace(f"%%MATHBLOCK_{i}%%", f'<div class="math-display">$${clean_b}$$</div>')

    for i, inl in enumerate(math_inlines):
        clean_inl = inl.strip()
        html_body = html_body.replace(f"%%MATHINLINE_{i}%%", f'${clean_inl}$')

    return html_body

def get_base_css(lang="vi"):
    return f'''
@page {{
    size: A4;
    margin: 22mm 18mm 22mm 18mm;
}}

body {{
    font-family: 'Times New Roman', Times, 'Cambria', serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #111827;
    text-align: justify;
    text-justify: inter-word;
    margin: 0;
    padding: 0;
}}

h1 {{
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    line-height: 1.35;
    margin-top: 10px;
    margin-bottom: 12px;
    color: #0F172A;
    text-transform: uppercase;
}}

.author-header {{
    text-align: center;
    font-size: 9.5pt;
    margin-bottom: 22px;
    line-height: 1.45;
    color: #334155;
}}

.author-header strong {{
    color: #0F172A;
}}

.abstract-box {{
    background-color: #F8FAFC;
    border-left: 4px solid #1E40AF;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
    padding: 14px 18px;
    margin: 18px 0 24px 0;
    font-size: 9.2pt;
    line-height: 1.45;
    border-radius: 4px;
}}

.abstract-title {{
    font-size: 10.5pt;
    font-weight: bold;
    color: #1E40AF;
    margin-top: 0;
    margin-bottom: 8px;
    text-transform: uppercase;
    border: none;
    padding: 0;
}}

h2 {{
    font-size: 11.5pt;
    font-weight: bold;
    color: #0F172A;
    border-bottom: 1.5px solid #CBD5E1;
    padding-bottom: 3px;
    margin-top: 24px;
    margin-bottom: 10px;
    page-break-after: avoid;
    text-transform: uppercase;
}}

h3 {{
    font-size: 10.5pt;
    font-weight: bold;
    color: #1E293B;
    margin-top: 16px;
    margin-bottom: 6px;
    page-break-after: avoid;
}}

h4 {{
    font-size: 10pt;
    font-weight: bold;
    font-style: italic;
    color: #334155;
    margin-top: 12px;
    margin-bottom: 4px;
    page-break-after: avoid;
}}

p {{
    margin-top: 0;
    margin-bottom: 8px;
    text-indent: 1.5em;
}}

p:first-of-type, .abstract-box p, .figure-caption, table p {{
    text-indent: 0;
}}

ul, ol {{
    margin-top: 4px;
    margin-bottom: 10px;
    padding-left: 24px;
}}

li {{
    margin-bottom: 4px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 16px 0;
    font-size: 8.5pt;
    line-height: 1.35;
    page-break-inside: avoid;
}}

th {{
    border-top: 2px solid #0F172A;
    border-bottom: 1.5px solid #0F172A;
    padding: 6px 6px;
    text-align: center;
    font-weight: bold;
    background-color: #F1F5F9;
    color: #0F172A;
}}

td {{
    border-bottom: 1px solid #E2E8F0;
    padding: 5px 6px;
    text-align: center;
}}

td:first-child, th:first-child {{
    text-align: left;
}}

tr:last-child td {{
    border-bottom: 2px solid #0F172A;
}}

tr:nth-child(even) td {{
    background-color: #F8FAFC;
}}

.figure-box {{
    text-align: center;
    margin: 18px 0;
    page-break-inside: avoid;
}}

.figure-box img {{
    height: auto;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    border-radius: 4px;
}}

.figure-caption {{
    font-size: 8.5pt;
    color: #334155;
    margin-top: 6px;
    line-height: 1.35;
    font-style: italic;
    text-align: center;
}}

.math-display {{
    text-align: center;
    margin: 10px 0;
    page-break-inside: avoid;
    overflow-x: auto;
}}

code {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.8pt;
    background-color: #F1F5F9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0F172A;
}}

pre {{
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 4px;
    padding: 10px 14px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8pt;
    line-height: 1.35;
    overflow-x: auto;
    page-break-inside: avoid;
}}

hr {{
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 18px 0;
}}
'''

def build_vietnamese_pdf():
    print("\n--- Building Vietnamese PDF Paper ---")
    md_path = os.path.join(ASSETS_DIR, "BAN_THAO_BAI_BAO_TIENG_VIET.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    b64_fig0 = img_to_base64(os.path.join(FIGURES_DIR, "fig0_pipeline_architecture.png"))
    b64_fig1 = img_to_base64(os.path.join(FIGURES_DIR, "fig1_confusion_matrix.png"))
    b64_fig2 = img_to_base64(os.path.join(FIGURES_DIR, "fig2_csi_clinical_scatter.png"))
    b64_fig3 = img_to_base64(os.path.join(FIGURES_DIR, "fig3_temporal_timeline.png"))

    fig0_html = f'''
    <div class="figure-box">
        <img src="{b64_fig0}" alt="Hình 1" style="max-width: 95%;">
        <div class="figure-caption"><strong>Hình 1:</strong> Sơ đồ khối kiến trúc tổng thể của hệ thống nhận diện và tự động chấm điểm tuân thủ lâm sàng WHO qua 4 giai đoạn liên hoàn.</div>
    </div>
    '''

    fig1_html = f'''
    <div class="figure-box">
        <img src="{b64_fig1}" alt="Hình 2" style="max-width: 78%;">
        <div class="figure-caption"><strong>Hình 2:</strong> Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix) của mô hình BiLSTM + Temporal Attention trên tập Test độc lập (3.638 cửa sổ dữ liệu).</div>
    </div>
    '''

    fig2_html = f'''
    <div class="figure-box">
        <img src="{b64_fig2}" alt="Hình 3" style="max-width: 76%;">
        <div class="figure-caption"><strong>Hình 3:</strong> Tương quan phân tán (Clinical Scatter Plot) giữa Điểm tuân thủ lâm sàng thực tế của Kiểm toán viên Y tế và Điểm số dự đoán tự động của Hệ thống AI ($r = 0.7568$, $p = 4.34 \\times 10^{{-9}}$).</div>
    </div>
    '''

    fig3_html = f'''
    <div class="figure-box">
        <img src="{b64_fig3}" alt="Hình 4" style="max-width: 96%;">
        <div class="figure-caption"><strong>Hình 4:</strong> Minh họa tiến trình thời gian thực (Dense Temporal Timeline) dự đoán liên tục 6 bước WHO của một video kiểm thử độc lập đối chiếu với nhãn Ground Truth lâm sàng.</div>
    </div>
    '''

    ascii_pattern = re.compile(r'```.*?\[Video Camera\].*?```', re.DOTALL)
    if ascii_pattern.search(md_text):
        md_text = ascii_pattern.sub("<!-- FIG0_PLACEHOLDER -->", md_text)

    html_body = render_math_safe_markdown(md_text)

    if "<!-- FIG0_PLACEHOLDER -->" in html_body:
        html_body = html_body.replace("<!-- FIG0_PLACEHOLDER -->", fig0_html)
    else:
        html_body = html_body.replace("<h2>3. PHƯƠNG PHÁP ĐỀ XUẤT", f"{fig0_html}\n<h2>3. PHƯƠNG PHÁP ĐỀ XUẤT")

    table3_marker = "Bảng 3:"
    if table3_marker in html_body:
        parts = html_body.split(table3_marker, 1)
        t_end = parts[1].find("</table>")
        if t_end != -1:
            idx = t_end + len("</table>")
            parts[1] = parts[1][:idx] + "\n" + fig1_html + parts[1][idx:]
            html_body = parts[0] + table3_marker + parts[1]

    table4_marker = "Bảng 4:"
    if table4_marker in html_body:
        parts = html_body.split(table4_marker, 1)
        t_end = parts[1].find("</table>")
        if t_end != -1:
            idx = t_end + len("</table>")
            parts[1] = parts[1][:idx] + "\n" + fig2_html + parts[1][idx:]
            html_body = parts[0] + table4_marker + parts[1]

    table5_marker = "Bảng 5:"
    if table5_marker in html_body:
        parts = html_body.split(table5_marker, 1)
        t_end = parts[1].find("</table>")
        if t_end != -1:
            idx = t_end + len("</table>")
            parts[1] = parts[1][:idx] + "\n" + fig3_html + parts[1][idx:]
            html_body = parts[0] + table5_marker + parts[1]

    html_body = re.sub(
        r'<h2>TÓM TẮT \(ABSTRACT\)</h2>\s*<p>(.*?)</p>',
        r'<div class="abstract-box"><h2 class="abstract-title">TÓM TẮT (ABSTRACT)</h2><p>\1</p></div>',
        html_body,
        flags=re.DOTALL
    )

    full_html = f'''<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Nhận diện Động tác Rửa tay Chuẩn WHO - AI Healthcare</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
        ],
        throwOnError: false
    }});"></script>
<style>
{get_base_css("vi")}
</style>
</head>
<body>
{html_body}
</body>
</html>'''

    html_out = os.path.join(ASSETS_DIR, "BAI_BAO_KHOA_HOC_WHO_HANDWASHING_TIENG_VIET.html")
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(full_html)

    pdf_out = os.path.join(ASSETS_DIR, "BAI_BAO_KHOA_HOC_WHO_HANDWASHING_TIENG_VIET.pdf")
    header_tpl = '<div style="font-size: 8pt; width: 100%; text-align: right; color: #64748B; padding-right: 18mm; font-family: Times New Roman, serif; font-style: italic;">Nghiên cứu Nhận diện Động tác Rửa tay Chuẩn WHO - AI Healthcare</div>'
    footer_tpl = '<div style="font-size: 8.5pt; width: 100%; text-align: center; color: #475569; font-family: Times New Roman, serif;">Trang <span class="pageNumber"></span> / <span class="totalPages"></span></div>'

    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=8000",
        "--display-header-footer",
        f"--header-template={header_tpl}",
        f"--footer-template={footer_tpl}",
        f"--print-to-pdf={pdf_out}",
        html_out
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if os.path.exists(pdf_out):
        size_mb = os.path.getsize(pdf_out) / (1024 * 1024)
        print(f"SUCCESS (VI): Generated {pdf_out} ({size_mb:.2f} MB)")
    else:
        print("ERROR: Vietnamese PDF was not generated!")

def build_english_pdf():
    print("\n--- Building English PDF Manuscript ---")
    md_path = os.path.join(ASSETS_DIR, "MANUSCRIPT.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    b64_fig0 = img_to_base64(os.path.join(FIGURES_DIR, "fig0_pipeline_architecture.png"))
    b64_fig1 = img_to_base64(os.path.join(FIGURES_DIR, "fig1_confusion_matrix.png"))
    b64_fig2 = img_to_base64(os.path.join(FIGURES_DIR, "fig2_csi_clinical_scatter.png"))
    b64_fig3 = img_to_base64(os.path.join(FIGURES_DIR, "fig3_temporal_timeline.png"))

    fig0_html = f'''
    <div class="figure-box">
        <img src="{b64_fig0}" alt="Figure 1" style="max-width: 95%;">
        <div class="figure-caption"><strong>Figure 1:</strong> End-to-end multi-stage architecture of the proposed WHO hand hygiene recognition and automated clinical compliance scoring system.</div>
    </div>
    '''

    fig1_html = f'''
    <div class="figure-box">
        <img src="{b64_fig1}" alt="Figure 2" style="max-width: 78%;">
        <div class="figure-caption"><strong>Figure 2:</strong> Normalized confusion matrix of the proposed BiLSTM-Attn model on the independent test split (3,638 sliding windows).</div>
    </div>
    '''

    fig2_html = f'''
    <div class="figure-box">
        <img src="{b64_fig2}" alt="Figure 3" style="max-width: 76%;">
        <div class="figure-caption"><strong>Figure 3:</strong> Clinical correlation scatter plot between ground truth expert compliance scores and AI-predicted Compliance Scoring Index (CSI) across all 43 test videos ($r = 0.7568$, $p = 4.34 \\times 10^{{-9}}$).</div>
    </div>
    '''

    fig3_html = f'''
    <div class="figure-box">
        <img src="{b64_fig3}" alt="Figure 4" style="max-width: 96%;">
        <div class="figure-caption"><strong>Figure 4:</strong> Dense frame-by-frame temporal prediction timeline comparing predicted WHO steps against ground truth expert annotations in a representative test video.</div>
    </div>
    '''

    html_body = render_math_safe_markdown(md_text)

    # Insert Fig 0 into Section 3
    if "<h2>3. Methodology</h2>" in html_body:
        html_body = html_body.replace("<h2>3. Methodology</h2>", f"<h2>3. Methodology</h2>\n{fig0_html}")

    # Insert Fig 1 (Confusion Matrix) after Table 3
    table3_marker = "Table 3:"
    if table3_marker in html_body:
        parts = html_body.split(table3_marker, 1)
        t_end = parts[1].find("</table>")
        if t_end != -1:
            idx = t_end + len("</table>")
            parts[1] = parts[1][:idx] + "\n" + fig1_html + parts[1][idx:]
            html_body = parts[0] + table3_marker + parts[1]

    # Insert Fig 2 (CSI Scatter Plot) after Table 4
    table4_marker = "Table 4:"
    if table4_marker in html_body:
        parts = html_body.split(table4_marker, 1)
        t_end = parts[1].find("</table>")
        if t_end != -1:
            idx = t_end + len("</table>")
            parts[1] = parts[1][:idx] + "\n" + fig2_html + parts[1][idx:]
            html_body = parts[0] + table4_marker + parts[1]

    # Insert Fig 3 (Timeline) after Table 5
    table5_marker = "Table 5:"
    if table5_marker in html_body:
        parts = html_body.split(table5_marker, 1)
        t_end = parts[1].find("</table>")
        if t_end != -1:
            idx = t_end + len("</table>")
            parts[1] = parts[1][:idx] + "\n" + fig3_html + parts[1][idx:]
            html_body = parts[0] + table5_marker + parts[1]

    # Wrap Abstract in abstract-box
    html_body = re.sub(
        r'<h2>Abstract</h2>\s*<p>(.*?)</p>',
        r'<div class="abstract-box"><h2 class="abstract-title">Abstract</h2><p>\1</p></div>',
        html_body,
        flags=re.DOTALL
    )

    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Geometric-Invariant Skeleton Dynamics for WHO Hand Hygiene Protocol Auditing</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
        ],
        throwOnError: false
    }});"></script>
<style>
{get_base_css("en")}
</style>
</head>
<body>
{html_body}
</body>
</html>'''

    html_out = os.path.join(ASSETS_DIR, "MANUSCRIPT_WHO_HANDWASHING_ENGLISH.html")
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(full_html)

    pdf_out = os.path.join(ASSETS_DIR, "MANUSCRIPT_WHO_HANDWASHING_ENGLISH.pdf")
    header_tpl = '<div style="font-size: 8pt; width: 100%; text-align: right; color: #64748B; padding-right: 18mm; font-family: Times New Roman, serif; font-style: italic;">Geometric-Invariant Skeleton Dynamics for WHO Hand Hygiene Auditing</div>'
    footer_tpl = '<div style="font-size: 8.5pt; width: 100%; text-align: center; color: #475569; font-family: Times New Roman, serif;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'

    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=8000",
        "--display-header-footer",
        f"--header-template={header_tpl}",
        f"--footer-template={footer_tpl}",
        f"--print-to-pdf={pdf_out}",
        html_out
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if os.path.exists(pdf_out):
        size_mb = os.path.getsize(pdf_out) / (1024 * 1024)
        print(f"SUCCESS (EN): Generated {pdf_out} ({size_mb:.2f} MB)")
    else:
        print("ERROR: English PDF was not generated!")

if __name__ == "__main__":
    build_vietnamese_pdf()
    build_english_pdf()
