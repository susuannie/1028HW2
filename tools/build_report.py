"""把 README.md 轉成報告書 PDF（notebooks/HW2_report.pdf）。

用法：
    python tools/build_report.py                 # 產生 HTML，並嘗試用 Edge / Chrome 轉成 PDF
    python tools/build_report.py --author "姓名 學號"

找不到瀏覽器時，會留下 notebooks/HW2_report.html，用瀏覽器開啟後「列印 → 另存為 PDF」即可。
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OUT_HTML = ROOT / "notebooks" / "HW2_report.html"
OUT_PDF = ROOT / "notebooks" / "HW2_report.pdf"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "chromium", "msedge",
]

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Microsoft JhengHei", "Noto Sans TC", "PingFang TC", sans-serif; color: #3b2e25;
       font-size: 10.5pt; line-height: 1.6; }
.cover { border-bottom: 3px solid #8c6240; padding-bottom: 10px; margin-bottom: 14px; }
.cover .course { color: #8a7665; font-size: 10pt; }
.cover .meta { color: #8a7665; font-size: 10pt; margin-top: 4px; }
h1 { color: #6b4a33; font-size: 20pt; margin: 4px 0; }
h2 { color: #6b4a33; font-size: 14.5pt; border-bottom: 1px solid #e7dccd; padding-bottom: 3px; margin-top: 22px;
     break-after: avoid; }
h3 { color: #8c6240; font-size: 12pt; margin-top: 14px; break-after: avoid; }
a { color: #8c6240; }
blockquote { margin: 6px 0; padding: 6px 12px; background: #faf5ee; border-left: 4px solid #c9923f; color: #5f4b3c; }
blockquote p { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; break-inside: avoid; }
th, td { border: 1px solid #e7dccd; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #f3eadf; }
img { max-width: 100%; display: block; margin: 8px auto; border: 1px solid #e7dccd; border-radius: 4px; break-inside: avoid; }
code { background: #f3eadf; padding: 0 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #faf5ee; padding: 8px 10px; border-radius: 6px; font-size: 9pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
hr { border: 0; border-top: 1px dashed #e7dccd; margin: 18px 0; }
"""


def build_html(author):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", text, flags=re.M).group(1)
    text = re.sub(r"^# .+\n", "", text, count=1, flags=re.M)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # 讓圖片在 notebooks/ 底下的 HTML 也找得到
    body = re.sub(r'src="(?!https?:|file:)([^"]+)"', lambda m: f'src="{(ROOT / m.group(1)).as_uri()}"', body)
    meta = f"{author} · " if author else ""
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style></head><body>
<div class="cover"><div class="course">AIoT 物聯網數據分析與應用 · HW2 Multiple Linear Regression</div>
<h1>{title}</h1><div class="meta">{meta}報告日期：{date.today():%Y-%m-%d}</div></div>
{body}</body></html>"""


def find_browser():
    for b in BROWSERS:
        path = shutil.which(b) or (b if Path(b).exists() else None)
        if path:
            return path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="", help="封面顯示的姓名／學號")
    args = ap.parse_args()

    OUT_HTML.write_text(build_html(args.author), encoding="utf-8")
    print(f"HTML: {OUT_HTML}")

    browser = find_browser()
    if not browser:
        print("找不到 Edge / Chrome：請用瀏覽器開啟上面的 HTML，列印 → 另存為 PDF。")
        return
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print(f"PDF:  {OUT_PDF}")


if __name__ == "__main__":
    main()
