#!/usr/bin/env python3
"""Local ALTERSENSE operator KPI dashboard."""

from __future__ import annotations

import argparse
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def pct_class(value: float) -> str:
    if value >= 80:
        return "good"
    if value >= 60:
        return "mid"
    return "low"


def reliability_class(value: str) -> str:
    mapping = {
        "validated": "good",
        "partial": "mid",
        "unreliable": "low",
    }
    return mapping.get(value, "mid")


def html_page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #eef5ff;
      --paper: #ffffff;
      --ink: #16314a;
      --muted: #668099;
      --line: #d2dfef;
      --accent: #1d66d8;
      --accent-dark: #0e458f;
      --soft: #f7fbff;
      --good: #20a85b;
      --warn: #ea9a12;
      --bad: #d64f4f;
      --shadow: 0 18px 36px rgba(18, 49, 84, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(29,102,216,0.12), transparent 30%),
        linear-gradient(180deg, #f8fbff, var(--bg));
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 24px 18px 48px; }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .mark {{
      width: 54px;
      height: 54px;
      border-radius: 16px;
      background: linear-gradient(140deg, #0d4ba2, #5fa7ff);
      display: grid;
      place-items: center;
      color: white;
      font-weight: 700;
      letter-spacing: 0.08em;
      box-shadow: var(--shadow);
    }}
    .brand strong {{ display: block; font-size: 1.22rem; }}
    .brand span, .muted {{ color: var(--muted); }}
    .hero {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 26px;
      padding: 28px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      align-items: center;
    }}
    .headline {{
      font-size: 2.2rem;
      line-height: 1.05;
      margin: 0 0 10px;
      letter-spacing: -0.03em;
    }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    .chip {{
      border-radius: 999px;
      padding: 8px 12px;
      background: #e9f3ff;
      color: var(--accent-dark);
      font-weight: 700;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .stat {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.8rem;
      color: var(--accent-dark);
    }}
    .layout-title {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin: 18px 0 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card-head {{
      padding: 20px 22px 14px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff, #f9fbff);
    }}
    .card-head h3 {{
      margin: 0;
      font-size: 1.34rem;
    }}
    .sub {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 10px;
      color: var(--muted);
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #edf5ff;
      color: var(--accent-dark);
      font-weight: 700;
      font-size: 0.93rem;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0;
      border-bottom: 1px solid var(--line);
    }}
    .kpi {{
      padding: 16px 18px;
      text-align: center;
      border-right: 1px solid var(--line);
    }}
    .kpi:last-child {{ border-right: none; }}
    .kpi label {{
      display: block;
      color: var(--muted);
      margin-bottom: 6px;
      font-size: 0.94rem;
    }}
    .kpi strong {{ font-size: 1.35rem; }}
    .good strong {{ color: var(--good); }}
    .mid strong {{ color: var(--warn); }}
    .low strong {{ color: var(--bad); }}
    .body {{
      padding: 18px 22px 22px;
      display: grid;
      gap: 16px;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .meta {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .meta label {{
      display: block;
      color: var(--muted);
      margin-bottom: 6px;
      font-size: 0.92rem;
    }}
    .mix {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .tag {{
      border-radius: 999px;
      background: #f0f6ff;
      border: 1px solid #d6e4f7;
      padding: 8px 12px;
      font-size: 0.92rem;
    }}
    @media (max-width: 1040px) {{
      .hero-grid, .cards {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .meta-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">{body}</div>
</body>
</html>""".encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    report_path: Path

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        totals = report["totals"]
        cards = []
        for station in report["stations"]:
            mix = "".join(
                f"<span class='tag'>{html.escape(label.replace('_', ' ').title())}: {html.escape(value)}</span>"
                for label, value in station.get("label_durations_human", {}).items()
            )
            logic_badge = "Postprocessed Cycle Logic" if station.get("current_cycle_label") else "Direct Activity Logic"
            cards.append(
                f"""
                <section class="card">
                  <div class="card-head">
                    <h3>{html.escape(station['operator_name'])}</h3>
                    <div class="sub">
                      <span>Station {html.escape(station['station_id'])}</span>
                      <span>{html.escape(station['employee_id'])}</span>
                      <span>{html.escape(station['operation_name'])}</span>
                      <span class="pill">{html.escape(station['station_role'].replace('_', ' ').title())}</span>
                      <span class="pill">{html.escape(logic_badge)}</span>
                      <span class="pill {reliability_class(station.get('reliability_badge', 'partial'))}">{html.escape(station.get('reliability_badge', 'partial').title())}</span>
                    </div>
                  </div>
                  <div class="kpis">
                    <div class="kpi {pct_class(station['efficiency_pct'])}">
                      <label>Efficiency</label>
                      <strong>{station['efficiency_pct']:.1f}%</strong>
                    </div>
                    <div class="kpi {pct_class(station['presence_pct'])}">
                      <label>Presence</label>
                      <strong>{station['presence_pct']:.1f}%</strong>
                    </div>
                    <div class="kpi">
                      <label>Verified Cycles</label>
                      <strong>{station.get('verified_cycle_count', station['cycle_count'])}</strong>
                    </div>
                    <div class="kpi">
                      <label>Avg Cycle</label>
                      <strong>{html.escape(station['duration_human']['avg_cycle'])}</strong>
                    </div>
                  </div>
                  <div class="body">
                    <div class="meta-grid">
                      <div class="meta">
                        <label>Present Time</label>
                        <strong>{html.escape(station['duration_human']['present'])}</strong>
                      </div>
                      <div class="meta">
                        <label>Working Time</label>
                        <strong>{html.escape(station['duration_human']['working'])}</strong>
                      </div>
                      <div class="meta">
                        <label>Productive Time</label>
                        <strong>{html.escape(station['duration_human'].get('productive', station['duration_human']['working']))}</strong>
                      </div>
                    </div>
                    <div class="meta-grid">
                      <div class="meta">
                        <label>Support Time</label>
                        <strong>{html.escape(station['duration_human'].get('support', '0m 00s'))}</strong>
                      </div>
                      <div class="meta">
                        <label>NPT Time</label>
                        <strong>{html.escape(station['duration_human']['npt'])}</strong>
                      </div>
                      <div class="meta">
                        <label>Presence Loss</label>
                        <strong>{html.escape(station['duration_human']['absent'])}</strong>
                      </div>
                    </div>
                    <div class="meta-grid">
                      <div class="meta">
                        <label>Observed Window</label>
                        <strong>{html.escape(str(round(station['observed_duration_sec'], 1)))} sec</strong>
                      </div>
                      <div class="meta">
                        <label>Sample Count</label>
                        <strong>{html.escape(str(station['sample_count']))}</strong>
                      </div>
                      <div class="meta">
                        <label>Latest Activity</label>
                        <strong>{html.escape(station.get('current_cycle_label', station['current_label']).replace('_', ' ').title())}</strong>
                      </div>
                      <div class="meta">
                        <label>Heuristic Cycles</label>
                        <strong>{html.escape(str(station.get('heuristic_cycle_count', station['cycle_count'])))}</strong>
                      </div>
                      <div class="meta">
                        <label>Heuristic Avg Cycle</label>
                        <strong>{html.escape(station['duration_human'].get('heuristic_avg_cycle', station['duration_human']['avg_cycle']))}</strong>
                      </div>
                      <div class="meta">
                        <label>Role Mismatches</label>
                        <strong>{html.escape(str(station.get('role_mismatch_count', 0)))}</strong>
                      </div>
                      <div class="meta">
                        <label>Open Cycle Candidates</label>
                        <strong>{html.escape(str(len(station.get('open_cycle_candidates', []))))}</strong>
                      </div>
                      <div class="meta">
                        <label>Reliability</label>
                        <strong>{html.escape(station.get('reliability_badge', 'partial').title())}</strong>
                      </div>
                    </div>
                    <div>
                      <div class="muted" style="margin-bottom:8px;">Activity Mix</div>
                      <div class="mix">{mix}</div>
                    </div>
                  </div>
                </section>
                """
            )

        body = f"""
        <section class="topbar">
          <div class="brand">
            <div class="mark">AS</div>
            <div>
              <strong>{html.escape(report['company_name'])}</strong>
              <span class="muted">{html.escape(report.get('factory_name', ''))} · {html.escape(report.get('floor_name', ''))} · {html.escape(report.get('line_name', ''))}</span>
            </div>
          </div>
          <div class="muted">Generated: {html.escape(report['generated_at_utc'])}</div>
        </section>
        <section class="hero">
          <div class="hero-grid">
            <div>
              <h1 class="headline">Operator productivity dashboard for station-wise worker activity and presence</h1>
              <div class="muted">This ALTERSENSE board converts workstation monitoring into operator KPIs: presence, productive working time, NPT, cycle count, and efficiency for each station.</div>
              <div class="chips">
                <span class="chip">Video: {html.escape(report.get('video_id', ''))}</span>
                <span class="chip">Stations: {totals['stations']}</span>
                <span class="chip">Cycle logic: verified + heuristic audit</span>
              </div>
            </div>
            <div class="summary">
              <div class="stat">
                <span class="muted">Total Present Time</span>
                <strong>{html.escape(totals['duration_human']['present'])}</strong>
              </div>
              <div class="stat">
                <span class="muted">Total Working Time</span>
                <strong>{html.escape(totals['duration_human']['working'])}</strong>
              </div>
              <div class="stat">
                <span class="muted">Total NPT</span>
                <strong>{html.escape(totals['duration_human']['npt'])}</strong>
              </div>
              <div class="stat">
                <span class="muted">Line Efficiency</span>
                <strong>{totals['efficiency_pct']:.1f}%</strong>
              </div>
              <div class="stat">
                <span class="muted">Verified Cycles</span>
                <strong>{totals.get('verified_cycle_count', totals['cycle_count'])}</strong>
              </div>
              <div class="stat">
                <span class="muted">Heuristic Cycles</span>
                <strong>{totals.get('heuristic_cycle_count', totals['cycle_count'])}</strong>
              </div>
            </div>
          </div>
        </section>
        <section class="layout-title">
          <div>
            <h2 style="margin:0;">Operator Layout</h2>
            <div class="muted">Station-by-station KPI cards with no graphs, focused on the exact numbers you asked for.</div>
          </div>
          <div class="pill">Verified Cycles: {totals.get('verified_cycle_count', totals['cycle_count'])} / Heuristic Cycles: {totals.get('heuristic_cycle_count', totals['cycle_count'])}</div>
        </section>
        <section class="cards">
          {''.join(cards)}
        </section>
        """
        payload = html_page("ALTERSENSE Operator Dashboard", body)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", default="artifacts/altersense/operator_report_cam33_clip_station_1456_station46_len12.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()

    DashboardHandler.report_path = Path(args.report_json).resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"ALTERSENSE operator dashboard running at http://{args.host}:{args.port}")
    print("Use Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
