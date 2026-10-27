"""把 Claude Code 的對話紀錄（~/.claude/projects/<專案>/<session>.jsonl）轉成 PDF。

用法：
    python tools/export_conversation.py                # 節錄版：只保留 EXCERPT_KEEP 列出的專業討論
    python tools/export_conversation.py --full         # 完整版：全部對話 + 工具呼叫摘要
    python tools/export_conversation.py --full <path.jsonl>

輸出 notebooks/Claude_conversation.pdf。對話內容逐字保留、不改寫；
節錄版只整段省略未列入的對話，部分刪除處標示「（略）」。
"""
import html
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).parent))
from build_report import find_browser  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_HTML = ROOT / "notebooks" / "Claude_conversation.html"
OUT_PDF = ROOT / "notebooks" / "Claude_conversation.pdf"

MIDTURN = re.compile(r"The user sent a new message while you were working:\s*(.*?)\s*(?:This is how Claude Code surfaces|$)", re.S)
NOISE = re.compile(r"<(ide_opened_file|ide_selection|system-reminder)>.*?</\1>|\[Image: (original|source)[^\]]*\]", re.S)

# 節錄版保留的段落：以使用者訊息開頭比對，該則訊息與其後 Claude 的回覆一起保留
EXCERPT_KEEP = [
    "這是作業需求",
    "你是使用有20個特徵",
    "特徵選擇為什麼是10個",
    "部屬到vercel，有錯誤",
    "這些我們有做到嗎",
    "**資料集預測平均壽命方法摘要",
    "如果改做Spotify",
]
# 訊息內部分刪除（標示「（略）」），使用者與 Claude 的文字都適用
EXCERPT_REDACT = [
    (re.compile(r"，以及老師提供的同學範本github.*?regression"), "（略）"),
    (re.compile(r"，不是範本用的洪水資料集"), "（略）"),
]

# 紀錄檔未保存的 Claude 回覆（與工具呼叫同一次輸出時，文字未寫入 .jsonl），依對話畫面逐字補錄
SUPPLEMENT_NOTE = "（此段回覆紀錄檔未保存，依對話畫面補錄）"
SUPPLEMENT = {
    "你是使用有20個特徵": """對，我用的是 Kaggle 上的 **Life Expectancy (WHO)**（kumarajarshi），不是範本用的洪水資料集。這份資料共 22 欄：

- 1 欄目標：`Life expectancy`
- 1 欄國名 `Country`：只當識別碼，用來依國家切分資料，不放進模型
- 其餘 **20 欄是特徵**：Year、Status（轉成 0/1）、Adult Mortality、HIV/AIDS、GDP、Schooling 等

20 個特徵大於 13，符合作業要求。接下來把 Notebook 轉檔並執行：""",
}
# Claude 回覆中含這些字的段落整段省略
EXCERPT_SKIP_REPLY = ("範本", "g115064011")

TOOL_LABEL = {
    "PowerShell": "執行指令", "Bash": "執行指令", "Write": "建立檔案", "Edit": "修改檔案", "Read": "讀取檔案",
    "Glob": "搜尋檔案", "Grep": "搜尋內容", "WebFetch": "讀取網頁", "WebSearch": "網路搜尋", "ToolSearch": "載入工具",
}

CSS = """
@page { size: A4; margin: 14mm 13mm; }
body { font-family: "Microsoft JhengHei", "Noto Sans TC", "PingFang TC", sans-serif; color: #3b2e25; font-size: 10pt; line-height: 1.55; }
h1 { color: #6b4a33; font-size: 18pt; margin: 0 0 4px; }
.meta { color: #8a7665; font-size: 9.5pt; border-bottom: 3px solid #8c6240; padding-bottom: 8px; margin-bottom: 14px; }
.msg { margin: 10px 0; padding: 8px 12px; border-radius: 8px; break-inside: auto; }
.user { background: #f1e4d4; border-left: 4px solid #8c6240; }
.claude { background: #fffcf7; border: 1px solid #e7dccd; }
.who { font-weight: 700; font-size: 9.5pt; margin-bottom: 2px; }
.user .who { color: #8c6240; } .claude .who { color: #6f7f45; }
.tool { color: #8a7665; font-size: 8.5pt; font-family: Consolas, monospace; margin: 1px 0; }
.attach { color: #8a7665; font-style: italic; }
table { border-collapse: collapse; margin: 6px 0; font-size: 9pt; }
th, td { border: 1px solid #e7dccd; padding: 3px 6px; }
th { background: #f3eadf; }
pre { background: #faf5ee; padding: 6px 8px; border-radius: 4px; font-size: 8.5pt; white-space: pre-wrap; }
code { background: #f3eadf; padding: 0 3px; border-radius: 3px; font-size: 9pt; }
p { margin: 4px 0; }
"""


def find_log():
    proj = Path.home() / ".claude" / "projects"
    logs = sorted(proj.glob("*HW2-Multiple-Linear-regression*/*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        sys.exit("找不到對話紀錄，請把 .jsonl 路徑當參數傳入。")
    return logs[-1]


def clean_user_text(text):
    m = MIDTURN.search(text)
    if m:
        text = m.group(1)
    return NOISE.sub("", text).strip()


def tool_summary(block):
    name, inp = block.get("name", ""), block.get("input", {})
    detail = inp.get("description") or inp.get("file_path") or inp.get("url") or inp.get("query") or inp.get("pattern") or ""
    detail = re.sub(r".*[\\/](?=[^\\/]+$)", "", detail) if "file_path" in inp and detail == inp["file_path"] else detail
    return f"🔧 {TOOL_LABEL.get(name, name)}：{detail}"


def parse(log_path):
    """回傳 [(role, kind, content)]，kind = text / tool / attach / supplement。"""
    items, pending = [], []  # pending：圖片常先於文字寫入紀錄檔，暫存後接在下一則使用者文字之後

    def add_user_text(txt):
        items.append(("user", "text", txt))
        items.extend(pending)
        pending.clear()
        for prefix, reply in SUPPLEMENT.items():
            if txt.startswith(prefix):
                items.append(("assistant", "supplement", reply))

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            role = o.get("type")
            att = o.get("attachment") or {}
            if role == "attachment" and att.get("type") == "queued_command" and att.get("humanTurn"):
                # 工作中途使用者送出的訊息
                for p in att.get("prompt") or []:
                    if p.get("type") == "text" and p["text"].strip():
                        add_user_text(p["text"].strip())
                continue
            if role not in ("user", "assistant") or o.get("isMeta"):
                continue
            content = (o.get("message") or {}).get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for b in content or []:
                t = b.get("type")
                if role == "user":
                    if t == "text":
                        txt = clean_user_text(b["text"])
                        if txt:
                            add_user_text(txt)
                    elif t == "image":
                        pending.append(("user", "attach", "[附圖]"))
                    elif t == "document":
                        pending.append(("user", "attach", "[附件：PDF 文件]"))
                elif t == "text" and b["text"].strip():
                    items.append(("assistant", "text", b["text"].strip()))
                elif t == "tool_use":
                    items.append(("assistant", "tool", tool_summary(b)))
    return items


def excerpt(items):
    """只留 EXCERPT_KEEP 的段落，去掉工具摘要與提到範本的回覆。"""
    out, keep = [], False
    for role, kind, content in items:
        if role == "user" and kind == "text":
            keep = any(content.startswith(p) for p in EXCERPT_KEEP)
        for pat, rep in EXCERPT_REDACT:
            content = pat.sub(rep, content)
        if not keep or kind == "tool":
            continue
        if role == "assistant" and any(s in content for s in EXCERPT_SKIP_REPLY):
            continue
        out.append((role, kind, content))
    return out


def render(items, log_path, is_excerpt=False):
    out, cur = [], None
    for role, kind, content in items:
        if role != cur:
            if cur:
                out.append("</div>")
            who = "👤 使用者" if role == "user" else "🤖 Claude"
            out.append(f'<div class="msg {"user" if role == "user" else "claude"}"><div class="who">{who}</div>')
            cur = role
        if kind == "tool":
            out.append(f'<div class="tool">{html.escape(content)}</div>')
        elif kind == "attach":
            out.append(f'<div class="attach">{content}</div>')
        elif role == "user":
            out.append("<p>" + html.escape(content).replace("\n", "<br>") + "</p>")
        elif kind == "supplement":
            out.append(f'<div class="attach">{SUPPLEMENT_NOTE}</div>')
            out.append(markdown.markdown(content, extensions=["tables", "fenced_code", "sane_lists"]))
        else:
            out.append(markdown.markdown(content, extensions=["tables", "fenced_code", "sane_lists"]))
    if cur:
        out.append("</div>")
    n_user = sum(1 for r, k, _ in items if r == "user" and k == "text")
    n_tool = sum(1 for _, k, _ in items if k == "tool")
    stamp = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    if is_excerpt:
        title = "Claude 對話紀錄（節錄）"
        note = (f"由 Claude Code 對話紀錄檔自動轉出之<b>節錄版</b>：保留與資料分析、建模、部署相關的 {n_user} 段專業討論，"
                "內容逐字保留未改寫；行政與操作性對話（檔案上傳步驟、版面配色等）整段省略，"
                "訊息內部分刪除處標示「（略）」；另省略工具執行過程與模型內部推理。")
    else:
        title = "Claude 對話紀錄"
        note = (f"由 Claude Code 對話紀錄檔自動轉出：完整保留使用者訊息（{n_user} 則）與 Claude 的文字回覆；"
                f"工具呼叫（{n_tool} 次）僅列一行摘要，省略工具輸出與模型內部推理。")
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style></head><body>
<h1>AIoT HW2 — {title}</h1>
<div class="meta">工具：Claude（Anthropic），透過 VS Code 的 Claude Code 擴充功能使用 · 紀錄時間至 {stamp}<br>{note}</div>
{''.join(out)}</body></html>"""


def main():
    args = [a for a in sys.argv[1:] if a != "--full"]
    full = "--full" in sys.argv
    log_path = Path(args[0]) if args else find_log()
    items = parse(log_path)
    if not full:
        items = excerpt(items)
    OUT_HTML.write_text(render(items, log_path, is_excerpt=not full), encoding="utf-8")
    browser = find_browser()
    if not browser:
        print(f"找不到 Edge / Chrome：請用瀏覽器開啟 {OUT_HTML}，列印 → 另存為 PDF。")
        return
    import subprocess
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    OUT_HTML.unlink()
    print(f"PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()
