"""
Lightweight HTTP monitoring dashboard for the ERP cache experiment.

Serves a single auto-refreshing HTML page on port 8080.
Start in its own tmux session:
  tmux new -d -s monitor "cd ~/erp_cache && source venv/bin/activate && python monitor_server.py"

Access from any device:  http://<VPS_IP>:8080
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8080
VPS_IP = os.environ.get("VPS_IP", "146.190.99.233")


def _read_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def _heartbeat_age_seconds(heartbeat_text: str) -> float | None:
    """Parse 'YYYY-MM-DD HH:MM:SS step=...' and return seconds since then."""
    if not heartbeat_text:
        return None
    try:
        ts_str = heartbeat_text.split(" step=")[0].strip()
        t = time.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return time.time() - time.mktime(t)
    except Exception:
        return None


def _completed_seeds_table(checkpoint: dict) -> str:
    seed_results = checkpoint.get("seed_results", [])
    if not seed_results:
        return "<p>No seeds completed yet.</p>"

    rows = ""
    for r in seed_results:
        seed = r.get("seed", "?")
        lru = r.get("lru_hit", 0)
        lfu = r.get("lfu_hit", 0)
        slru = r.get("slru_hit", 0)
        arc = r.get("arc_hit", 0)
        dqn = r.get("dqn_hit", 0)
        dqn_color = "#2e7d32" if dqn > lru else "#c62828"
        rows += (
            f"<tr>"
            f"<td>{seed}</td>"
            f"<td>{lru:.4f}</td>"
            f"<td>{lfu:.4f}</td>"
            f"<td style='color:#9C27B0'>{slru:.4f}</td>"
            f"<td>{arc:.4f}</td>"
            f"<td style='color:{dqn_color};font-weight:bold'>{dqn:.4f}</td>"
            f"<td style='color:{dqn_color}'>{dqn - lru:+.4f}</td>"
            f"</tr>\n"
        )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Seed</th><th>LRU</th><th>LFU</th>
          <th style='color:#9C27B0'>SLRU</th><th>ARC</th>
          <th>DQN</th><th>DQN vs LRU</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


def build_page() -> str:
    progress = _read_json("results/progress.json")
    checkpoint = _read_json("results/checkpoint.json")
    heartbeat_text = _read_text("results/heartbeat.txt")
    age = _heartbeat_age_seconds(heartbeat_text)

    # Status color
    if age is None:
        status_color = "#9e9e9e"
        status_label = "No heartbeat yet"
    elif age < 60:
        status_color = "#2e7d32"
        status_label = f"Running (last beat {int(age)}s ago)"
    elif age < 120:
        status_color = "#f57f17"
        status_label = f"Slow heartbeat ({int(age)}s ago)"
    else:
        status_color = "#c62828"
        status_label = f"Possible crash (last beat {int(age // 60)}min ago)"

    phase = progress.get("phase", "—")
    current = progress.get("current", 0)
    total = progress.get("total", 0)
    percent = progress.get("percent", 0.0)
    eta = progress.get("eta_human", "—")
    detail = progress.get("detail", "")
    timestamp = progress.get("timestamp", "—")
    elapsed_s = progress.get("elapsed_seconds", 0)
    elapsed_h = f"{int(elapsed_s)//3600}h {(int(elapsed_s)%3600)//60}m"

    completed_count = len(checkpoint.get("completed_seeds", []))
    last_completed = checkpoint.get("last_completed", "—")

    progress_bar_width = max(0, min(100, int(percent)))
    bar_color = "#1565c0" if phase == "main_experiment" else "#6a1b9a"

    seeds_table = _completed_seeds_table(checkpoint)

    # Compute live DQN mean if we have seed results
    seed_results = checkpoint.get("seed_results", [])
    if seed_results:
        dqn_hits = [r.get("dqn_hit", 0) for r in seed_results]
        lru_hits = [r.get("lru_hit", 0) for r in seed_results]
        dqn_mean = sum(dqn_hits) / len(dqn_hits)
        lru_mean = sum(lru_hits) / len(lru_hits)
        live_stats = (
            f"<p><b>Live DQN hit ratio mean:</b> {dqn_mean:.4f} "
            f"&nbsp;|&nbsp; <b>LRU mean:</b> {lru_mean:.4f} "
            f"&nbsp;|&nbsp; <b>Delta:</b> {dqn_mean - lru_mean:+.4f} "
            f"&nbsp;({completed_count} seeds so far)</p>"
        )
    else:
        live_stats = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <title>ERP Cache Experiment Monitor</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5; color: #212121; padding: 16px; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 16px; color: #1a237e; }}
    h2 {{ font-size: 1rem; margin: 20px 0 8px; color: #37474f; }}
    .card {{ background: #fff; border-radius: 8px; padding: 16px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.12); margin-bottom: 14px; }}
    .status-dot {{ display: inline-block; width: 12px; height: 12px;
                   border-radius: 50%; background: {status_color};
                   margin-right: 8px; vertical-align: middle; }}
    .status-label {{ font-weight: 600; color: {status_color}; vertical-align: middle; }}
    .progress-outer {{ background: #e0e0e0; border-radius: 6px;
                       height: 22px; margin: 10px 0; overflow: hidden; }}
    .progress-inner {{ background: {bar_color}; height: 100%;
                       width: {progress_bar_width}%; border-radius: 6px;
                       transition: width 0.5s ease;
                       display: flex; align-items: center; justify-content: center; }}
    .progress-text {{ color: #fff; font-size: 0.78rem; font-weight: 600;
                      white-space: nowrap; padding: 0 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap: 10px; }}
    .kv {{ background: #e8eaf6; border-radius: 6px; padding: 10px; }}
    .kv-label {{ font-size: 0.7rem; color: #5c6bc0; text-transform: uppercase;
                 letter-spacing: 0.04em; margin-bottom: 4px; }}
    .kv-value {{ font-size: 1.05rem; font-weight: 700; color: #1a237e; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #eeeeee; }}
    th {{ background: #e8eaf6; color: #1a237e; font-weight: 600; text-align: right; }}
    th:first-child, td:first-child {{ text-align: center; }}
    tr:last-child td {{ border-bottom: none; }}
    .refresh-note {{ font-size: 0.72rem; color: #9e9e9e; margin-top: 12px; text-align: right; }}
    .heartbeat {{ font-size: 0.8rem; color: #616161; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>ERP Cache Experiment Monitor</h1>

  <div class="card">
    <span class="status-dot"></span>
    <span class="status-label">{status_label}</span>
    <p class="heartbeat">Heartbeat: {heartbeat_text or "none yet"}</p>
  </div>

  <div class="card">
    <h2>Progress — {phase.replace("_", " ").title()}</h2>
    <div class="progress-outer">
      <div class="progress-inner">
        <span class="progress-text">{percent:.1f}% &nbsp; ({current}/{total})</span>
      </div>
    </div>
    <div class="grid">
      <div class="kv"><div class="kv-label">Seeds done</div>
        <div class="kv-value">{completed_count} / 20</div></div>
      <div class="kv"><div class="kv-label">Last completed</div>
        <div class="kv-value">Seed {last_completed}</div></div>
      <div class="kv"><div class="kv-label">ETA</div>
        <div class="kv-value">{eta}</div></div>
      <div class="kv"><div class="kv-label">Elapsed</div>
        <div class="kv-value">{elapsed_h}</div></div>
      <div class="kv"><div class="kv-label">Current step</div>
        <div class="kv-value">{detail}</div></div>
      <div class="kv"><div class="kv-label">Updated</div>
        <div class="kv-value" style="font-size:0.8rem">{timestamp}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Live Results ({completed_count} seeds)</h2>
    {live_stats}
    {seeds_table}
  </div>

  <p class="refresh-note">Auto-refreshes every 30 seconds &nbsp;&bull;&nbsp;
     Server time: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            body = build_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode())

    def log_message(self, fmt, *args):
        pass  # suppress per-request log noise


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    print(f"Monitor running at http://{VPS_IP}:{PORT}")
    print("Ctrl+C to stop.")
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
