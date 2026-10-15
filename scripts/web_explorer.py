#!/usr/bin/env python3
"""
DFlowNovo Web File System & Visual Plot Explorer
=================================================
A lightweight, modern web interface to navigate the project workspace,
inspect validation plots, PAUC curves, training logs, JSON metrics, and CSVs.
"""

import argparse
import html
import mimetypes
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="DFlowNovo Workspace Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TENSORBOARD_URL: Optional[str] = None


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable units."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def safe_path(rel_path: str) -> Path:
    """Resolve and enforce that the path remains within REPO_ROOT."""
    clean_rel = rel_path.lstrip("/").strip()
    target = (REPO_ROOT / clean_rel).resolve()
    try:
        target.relative_to(REPO_ROOT)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: outside workspace root.")
    return target


def get_breadcrumbs_html(rel_path: str) -> str:
    parts = [p for p in rel_path.strip("/").split("/") if p]
    crumbs = ['<a href="/browse" class="crumb-link">🏠 Root</a>']
    accum = []
    for part in parts:
        accum.append(part)
        subpath = "/".join(accum)
        crumbs.append(f'<span class="crumb-sep">/</span><a href="/browse?path={html.escape(subpath)}" class="crumb-link">{html.escape(part)}</a>')
    return "".join(crumbs)


def render_html_page(title: str, body_content: str, current_path: str = "") -> str:
    tb_btn = ""
    if TENSORBOARD_URL:
        tb_btn = f'''
        <a href="{TENSORBOARD_URL}" target="_blank" class="nav-btn tb-btn">
            📊 <strong>Open Live TensorBoard</strong> ↗
        </a>
        '''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)} - DFlowNovo Explorer</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --surface: #1e293b;
            --surface-hover: #334155;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-color);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            min-height: 100vh;
        }}
        header {{
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            text-decoration: none;
            color: var(--text);
            font-weight: 700;
            font-size: 1.25rem;
        }}
        .brand span.badge {{
            font-size: 0.75rem;
            background: #0284c7;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            font-weight: 600;
        }}
        .nav-actions {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}
        .nav-btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.5rem 0.9rem;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.15s ease;
            background: var(--surface-hover);
            color: var(--text);
            border: 1px solid var(--border);
        }}
        .nav-btn:hover {{
            background: #475569;
            border-color: #64748b;
        }}
        .tb-btn {{
            background: #0284c7;
            color: white;
            border-color: #38bdf8;
        }}
        .tb-btn:hover {{
            background: #0369a1;
        }}
        .quick-links {{
            background: #1e293b;
            border-bottom: 1px solid var(--border);
            padding: 0.5rem 2rem;
            display: flex;
            gap: 0.75rem;
            overflow-x: auto;
            white-space: nowrap;
        }}
        .ql-tag {{
            font-size: 0.8rem;
            color: var(--accent);
            text-decoration: none;
            padding: 0.25rem 0.6rem;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 4px;
        }}
        .ql-tag:hover {{
            background: rgba(56, 189, 248, 0.2);
        }}
        .main-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 1.5rem 2rem 3rem 2rem;
        }}
        .breadcrumbs {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.4rem;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
            background: var(--surface);
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .crumb-link {{
            color: var(--accent);
            text-decoration: none;
        }}
        .crumb-link:hover {{ text-decoration: underline; }}
        .crumb-sep {{ color: var(--text-muted); }}
        
        /* Plot Gallery Grid */
        .section-header {{
            font-size: 1.15rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}
        .gallery-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
        }}
        .gallery-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
            border-color: var(--accent);
        }}
        .gallery-img-container {{
            position: relative;
            background: #020617;
            padding: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 220px;
            cursor: pointer;
        }}
        .gallery-img-container img {{
            max-width: 100%;
            max-height: 260px;
            object-fit: contain;
            border-radius: 4px;
        }}
        .gallery-meta {{
            padding: 0.75rem 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .gallery-title {{
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text);
            text-decoration: none;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .gallery-actions {{
            display: flex;
            gap: 0.4rem;
        }}
        .btn-small {{
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            background: var(--surface-hover);
            color: var(--accent);
            text-decoration: none;
            border: 1px solid var(--border);
        }}
        .btn-small:hover {{
            background: #475569;
        }}

        /* Table View */
        .file-table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow-x: auto;
        }}
        table.file-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }}
        table.file-table th {{
            background: #1e293b;
            padding: 0.75rem 1rem;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
            font-weight: 600;
        }}
        table.file-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
            vertical-align: middle;
        }}
        table.file-table tr:hover td {{
            background: var(--surface-hover);
        }}
        .file-link {{
            color: var(--text);
            text-decoration: none;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .file-link:hover {{
            color: var(--accent);
        }}
        .dir-link {{
            color: var(--accent);
            font-weight: 600;
        }}

        /* Search input */
        .search-box {{
            padding: 0.5rem 1rem;
            background: var(--bg-color);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: white;
            font-size: 0.875rem;
            width: 250px;
        }}
        .search-box:focus {{
            outline: none;
            border-color: var(--accent);
        }}

        /* Lightbox Modal */
        #lightbox {{
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            padding: 2rem;
        }}
        #lightbox.active {{ display: flex; }}
        #lightbox img {{
            max-width: 90vw;
            max-height: 85vh;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        }}
        #lightbox-caption {{
            color: white;
            margin-top: 1rem;
            font-size: 1rem;
            font-weight: 500;
        }}
        .close-btn {{
            position: absolute;
            top: 1.5rem;
            right: 2rem;
            color: white;
            font-size: 2rem;
            cursor: pointer;
            background: none;
            border: none;
        }}

        /* Viewer Styles */
        .code-container {{
            background: #020617;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            overflow-x: auto;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            font-size: 0.875rem;
            line-height: 1.6;
            color: #e2e8f0;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .viewer-toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            background: var(--surface);
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <header>
        <a href="/browse" class="brand">
            🧬 DFlowNovo Explorer
            <span class="badge">Live</span>
        </a>
        <div class="nav-actions">
            {tb_btn}
            <a href="/browse?path=artifacts/canonical_reproduction" class="nav-btn">
                📁 Canonical Reproduction
            </a>
            <a href="/browse?path=artifacts/canonical_reproduction/stage1_pretrain/val_metrics" class="nav-btn">
                📈 Validation Curves
            </a>
            <a href="/browse?path=artifacts/canonical_reproduction/logs" class="nav-btn">
                📄 Training Logs
            </a>
        </div>
    </header>

    <div class="quick-links">
        <strong style="font-size: 0.8rem; color: var(--text-muted); align-self: center;">Quick Jumps:</strong>
        <a href="/browse?path=artifacts/canonical_reproduction" class="ql-tag">artifacts/canonical_reproduction</a>
        <a href="/browse?path=artifacts/canonical_reproduction/stage1_pretrain/val_metrics" class="ql-tag">stage1_pretrain/val_metrics</a>
        <a href="/browse?path=artifacts/canonical_reproduction/logs" class="ql-tag">stage_logs</a>
        <a href="/browse?path=paper/tables" class="ql-tag">paper/tables</a>
        <a href="/browse?path=scripts" class="ql-tag">scripts</a>
        <a href="/browse?path=src" class="ql-tag">src</a>
        <a href="/view?path=REPRODUCIBILITY.md" class="ql-tag">REPRODUCIBILITY.md</a>
    </div>

    <div class="main-container">
        {body_content}
    </div>

    <!-- Lightbox Modal -->
    <div id="lightbox" onclick="closeLightbox()">
        <button class="close-btn" onclick="closeLightbox()">&times;</button>
        <img id="lightbox-img" src="" alt="">
        <div id="lightbox-caption"></div>
    </div>

    <script>
        function openLightbox(src, caption) {{
            document.getElementById('lightbox-img').src = src;
            document.getElementById('lightbox-caption').innerText = caption;
            document.getElementById('lightbox').classList.add('active');
        }}
        function closeLightbox() {{
            document.getElementById('lightbox').classList.remove('active');
        }}
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') closeLightbox();
        }});

        function filterFiles() {{
            const query = document.getElementById('file-search').value.toLowerCase();
            const rows = document.querySelectorAll('table.file-table tbody tr');
            rows.forEach(r => {{
                const name = r.querySelector('.item-name').innerText.toLowerCase();
                r.style.display = name.includes(query) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(content='<script>window.location.href="/browse";</script>')


@app.get("/browse")
def browse(path: str = Query("", description="Relative path in repo")):
    target = safe_path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    if not target.is_dir():
        return HTMLResponse(content=f'<script>window.location.href="/view?path={html.escape(path)}";</script>')

    rel_str = str(target.relative_to(REPO_ROOT))
    if rel_str == ".":
        rel_str = ""

    breadcrumbs = get_breadcrumbs_html(rel_str)

    items = list(target.iterdir())
    items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

    # Separate directories, image files, and regular files
    dirs = []
    images = []
    files = []

    image_exts = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}

    for item in items:
        # Hide internal git directories if at root
        if item.name.startswith(".git") and rel_str == "":
            continue
        if item.is_dir():
            dirs.append(item)
        else:
            if item.suffix.lower() in image_exts:
                images.append(item)
            files.append(item)

    body = [f'<div class="breadcrumbs">{breadcrumbs}</div>']

    # Visual Plot Gallery (if images present)
    if images:
        body.append('<div class="section-header">')
        body.append(f'<span>📊 Visual Plot & Curve Gallery ({len(images)} figures)</span>')
        body.append('</div>')
        body.append('<div class="gallery-grid">')
        for img in images:
            img_rel = str(img.relative_to(REPO_ROOT))
            raw_url = f"/raw?path={html.escape(img_rel)}"
            view_url = f"/view?path={html.escape(img_rel)}"
            size_str = format_size(img.stat().st_size)
            body.append(f'''
            <div class="gallery-card">
                <div class="gallery-img-container" onclick="openLightbox('{raw_url}', '{html.escape(img.name)}')">
                    <img src="{raw_url}" alt="{html.escape(img.name)}" loading="lazy">
                </div>
                <div class="gallery-meta">
                    <a href="{view_url}" class="gallery-title" title="{html.escape(img.name)}">{html.escape(img.name)}</a>
                    <div class="gallery-actions">
                        <span style="font-size: 0.75rem; color: var(--text-muted); align-self: center;">{size_str}</span>
                        <a href="{raw_url}" download class="btn-small">⬇ Download</a>
                    </div>
                </div>
            </div>
            ''')
        body.append('</div>')

    # Directory and File Table
    body.append('<div class="section-header">')
    body.append(f'<span>📁 Directory Contents ({len(dirs)} folders, {len(files)} files)</span>')
    body.append('<input type="text" id="file-search" class="search-box" placeholder="Filter files..." onkeyup="filterFiles()">')
    body.append('</div>')

    body.append('<div class="file-table-container">')
    body.append('<table class="file-table">')
    body.append('<thead><tr><th>Name</th><th>Size</th><th>Last Modified</th><th>Action</th></tr></thead>')
    body.append('<tbody>')

    # Parent directory link
    if rel_str != "":
        parent_rel = str(target.parent.relative_to(REPO_ROOT))
        if parent_rel == ".":
            parent_rel = ""
        body.append(f'''
        <tr>
            <td><a href="/browse?path={html.escape(parent_rel)}" class="file-link dir-link">📁 .. (Parent Directory)</a></td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        ''')

    # Directories
    for d in dirs:
        d_rel = str(d.relative_to(REPO_ROOT))
        mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        num_children = len(list(d.iterdir())) if os.access(d, os.R_OK) else 0
        body.append(f'''
        <tr>
            <td><a href="/browse?path={html.escape(d_rel)}" class="file-link dir-link item-name">📁 {html.escape(d.name)}/</a></td>
            <td style="color: var(--text-muted);">{num_children} items</td>
            <td style="color: var(--text-muted);">{mtime}</td>
            <td><a href="/browse?path={html.escape(d_rel)}" class="btn-small">Open</a></td>
        </tr>
        ''')

    # Files
    for f in files:
        f_rel = str(f.relative_to(REPO_ROOT))
        size_str = format_size(f.stat().st_size)
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        icon = "📈" if f.suffix.lower() in image_exts else "📄" if f.suffix in [".json", ".csv", ".log", ".txt", ".md"] else "💾"
        body.append(f'''
        <tr>
            <td><a href="/view?path={html.escape(f_rel)}" class="file-link item-name">{icon} {html.escape(f.name)}</a></td>
            <td>{size_str}</td>
            <td style="color: var(--text-muted);">{mtime}</td>
            <td>
                <a href="/view?path={html.escape(f_rel)}" class="btn-small">View</a>
                <a href="/raw?path={html.escape(f_rel)}" download class="btn-small">Download</a>
            </td>
        </tr>
        ''')

    body.append('</tbody></table></div>')

    return HTMLResponse(content=render_html_page(f"Browsing /{rel_str}", "".join(body), rel_str))


@app.get("/view")
def view_file(path: str = Query(..., description="Relative path in repo")):
    target = safe_path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if target.is_dir():
        return HTMLResponse(content=f'<script>window.location.href="/browse?path={html.escape(path)}";</script>')

    rel_str = str(target.relative_to(REPO_ROOT))
    breadcrumbs = get_breadcrumbs_html(rel_str)
    size_str = format_size(target.stat().st_size)
    raw_url = f"/raw?path={html.escape(rel_str)}"

    image_exts = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}

    body = [f'<div class="breadcrumbs">{breadcrumbs}</div>']
    body.append(f'''
    <div class="viewer-toolbar">
        <div>
            <strong style="font-size: 1.1rem;">{html.escape(target.name)}</strong>
            <span style="color: var(--text-muted); margin-left: 0.75rem;">({size_str})</span>
        </div>
        <div>
            <a href="{raw_url}" download class="nav-btn">⬇ Download Raw</a>
            <a href="/browse?path={html.escape(str(target.parent.relative_to(REPO_ROOT)) if str(target.parent.relative_to(REPO_ROOT)) != '.' else '')}" class="nav-btn">📁 Open Containing Folder</a>
        </div>
    </div>
    ''')

    if target.suffix.lower() in image_exts:
        body.append(f'''
        <div style="background: #020617; border: 1px solid var(--border); border-radius: 8px; padding: 2rem; display: flex; justify-content: center; align-items: center; min-height: 500px;">
            <img src="{raw_url}" alt="{html.escape(target.name)}" style="max-width: 100%; max-height: 80vh; object-fit: contain; border-radius: 6px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
        </div>
        ''')
    elif target.suffix.lower() == ".json":
        import json
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            formatted = json.dumps(data, indent=2)
            body.append(f'<div class="code-container">{html.escape(formatted)}</div>')
        except Exception as e:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read(500000)
            body.append(f'<div class="code-container">{html.escape(raw_text)}</div>')
    elif target.suffix.lower() in [".txt", ".log", ".py", ".sh", ".tex", ".md", ".yaml", ".yml", ".jsonl"]:
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                # Read at most 1MB for preview
                content = f.read(1000000)
            body.append(f'<div class="code-container">{html.escape(content)}</div>')
        except Exception as e:
            body.append(f'<div class="code-container">Error reading file: {html.escape(str(e))}</div>')
    elif target.suffix.lower() == ".csv":
        # Parse CSV to table preview
        import csv
        try:
            rows = []
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 200:  # Preview first 200 rows
                        break
                    rows.append(row)
            if rows:
                body.append('<div class="file-table-container"><table class="file-table">')
                body.append('<thead><tr>')
                for col in rows[0]:
                    body.append(f'<th>{html.escape(col)}</th>')
                body.append('</tr></thead><tbody>')
                for row in rows[1:]:
                    body.append('<tr>')
                    for cell in row:
                        body.append(f'<td>{html.escape(cell)}</td>')
                    body.append('</tr>')
                body.append('</tbody></table></div>')
            else:
                body.append('<p style="color: var(--text-muted);">Empty CSV file.</p>')
        except Exception as e:
            body.append(f'<div class="code-container">Error rendering CSV: {html.escape(str(e))}</div>')
    else:
        # Binary file (checkpoint or other)
        body.append(f'''
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 3rem; text-align: center;">
            <p style="font-size: 1.2rem; margin-bottom: 1rem;">💾 Binary or Large Data File</p>
            <p style="color: var(--text-muted); margin-bottom: 1.5rem;">File size: {size_str}</p>
            <a href="{raw_url}" download class="nav-btn tb-btn">⬇ Download {html.escape(target.name)}</a>
        </div>
        ''')

    return HTMLResponse(content=render_html_page(target.name, "".join(body), rel_str))


@app.get("/raw")
def raw_file(path: str = Query(..., description="Relative path in repo")):
    target = safe_path(path)
    if not target.exists() or target.is_dir():
        raise HTTPException(status_code=404, detail="File not found")
    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "application/octet-stream", filename=target.name)


def main():
    parser = argparse.ArgumentParser(description="Run workspace file explorer and visual asset server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on.")
    parser.add_argument("--tensorboard-url", type=str, default=None, help="Public TensorBoard URL to link in header.")
    args = parser.parse_args()

    global TENSORBOARD_URL
    if args.tensorboard_url:
        TENSORBOARD_URL = args.tensorboard_url

    print(f"Starting DFlowNovo Explorer on http://{args.host}:{args.port}")
    if TENSORBOARD_URL:
        print(f"Linked TensorBoard: {TENSORBOARD_URL}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
