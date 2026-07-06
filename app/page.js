import { loadReport, summarizeReliability } from "@/lib/report";
import UploadEvaluationPanel from "@/app/components/upload-evaluation-panel";

export const dynamic = "force-dynamic";

function formatSeconds(seconds) {
  const total = Math.max(0, Math.round(Number(seconds || 0)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(secs).padStart(2, "0")}s`;
  }
  return `${minutes}m ${String(secs).padStart(2, "0")}s`;
}

function topCycleDisplay(report) {
  const status = report?.totals?.cycle_kpi?.status;
  if (status === "ok") {
    return report?.totals?.display_verified_cycle_count ?? report?.totals?.verified_cycle_count ?? 0;
  }
  return report?.totals?.verified_cycle_count ?? 0;
}

function initials(name) {
  return String(name || "")
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() || "")
    .join("");
}

function OperatorThumb({ station }) {
  if (station.station_image_url) {
    return (
      <img
        className="worker-avatar worker-avatar-image"
        src={station.station_image_url}
        alt={`${station.operator_name} top view`}
      />
    );
  }
  return <div className="worker-avatar">{initials(station.operator_name)}</div>;
}

const trustedBrands = ["Cutting", "Sewing", "Finishing", "Quality", "Output", "Operations"];
const deliveryPoints = ["Video upload", "Operator KPI board", "Overlay playback"];

function OperatorIdentity({ station }) {
  return (
    <div className="operator-identity">
      <div className="snapshot-meta">
        <span className="snapshot-label">Live Snapshot</span>
        <OperatorThumb station={station} />
      </div>
      <div className="operator-title-block">
        <h3>{station.operator_name}</h3>
        <p>
          Station {station.station_id} · {station.operation_name}
        </p>
        <span className="operator-id-chip">ID {station.employee_id}</span>
      </div>
    </div>
  );
}

function cycleDisplayCount(station) {
  if (station.station_role === "prep_pass") {
    return "N/A";
  }
  if (station.cycle_kpi_display?.count != null) {
    return station.cycle_kpi_display.count;
  }
  if ((station.verified_cycle_count ?? 0) > 0) {
    return station.verified_cycle_count;
  }
  return "Low confidence";
}

function cycleDisplayAverage(station) {
  if (station.station_role === "prep_pass") {
    return "N/A";
  }
  if (station.cycle_kpi_display?.status === "ok" && station.duration_human?.avg_cycle) {
    return station.duration_human.avg_cycle;
  }
  if ((station.verified_cycle_count ?? 0) > 0 && (station.average_cycle_time_sec ?? 0) > 0) {
    return formatSeconds(station.average_cycle_time_sec);
  }
  return "Low confidence";
}

function cycleDisplayClass(station) {
  return station.cycle_kpi_display?.status === "low_confidence" ? " muted-kpi" : "";
}

function utilizationSegments(station) {
  const present = Math.max(Number(station.present_duration_sec || 0), 1);
  const productive = Math.max(0, Number(station.productive_duration_sec || 0));
  const support = Math.max(0, Number(station.support_work_duration_sec || 0));
  const idle = Math.max(0, Number(station.npt_duration_sec || 0));
  return [
    { key: "productive", label: "Working", value: Math.min(100, (productive / present) * 100) },
    { key: "support", label: "Support", value: Math.min(100, (support / present) * 100) },
    { key: "idle", label: "Idle", value: Math.min(100, (idle / present) * 100) },
  ];
}

function UtilizationBar({ station }) {
  const segments = utilizationSegments(station);
  return (
    <div className="utilization-block">
      <div className="utilization-head">
        <label>Time Distribution</label>
      </div>
      <div className="utilization-bar" aria-hidden="true">
        {segments.map((segment) => (
          <span
            key={segment.key}
            className={`util-segment ${segment.key}`}
            style={{ width: `${Math.max(segment.value, segment.value > 0 ? 2 : 0)}%` }}
          />
        ))}
      </div>
      <div className="utilization-legend">
        {segments.map((segment) => (
          <span key={segment.key} className={`util-legend ${segment.key}`}>
            {segment.label}: {segment.value.toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  );
}

export default async function HomePage() {
  const report = await loadReport();
  const reliability = summarizeReliability(report.stations);

  return (
    <main className="page-shell">
      <section className="site-topbar">
        <div className="site-brand">
          <div className="brand-mark">OI</div>
          <div>
            <p className="eyebrow">Factory Performance</p>
            <strong className="brand-title">Operator Dashboard</strong>
          </div>
        </div>
        <nav className="site-nav" aria-label="Primary">
          <span>Overview</span>
          <span>Analysis</span>
          <span>Stations</span>
        </nav>
      </section>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Operations Monitoring</p>
          <h1>Upload a production-floor video and review operator performance in one clear dashboard</h1>
          <p className="hero-text">
            Review presence, working time, idle time, verified cycles, and station-level operator
            performance through a clean operational dashboard designed for factory use.
          </p>
          <div className="hero-chips">
            {deliveryPoints.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>
        <div className="hero-panel">
          <p className="eyebrow">Live Summary</p>
          <h2>Current Line Overview</h2>
          <p>Latest station report prepared for review and decision-making.</p>
          <div className="hero-panel-metrics">
            <div>
              <label>Verified Cycles</label>
              <strong>{topCycleDisplay(report)}</strong>
              <span>current reported total</span>
            </div>
            <div>
              <label>Tracked Operators</label>
              <strong>{report.stations.length}</strong>
              <span>active station cards</span>
            </div>
          </div>
          <div className="hero-runtime-list">
            <span>Overlay review available</span>
            <span>Station-level KPI reporting</span>
          </div>
        </div>
      </section>

      <section className="brand-strip">
        <div>
          <p className="eyebrow">Operations Scope</p>
          <h2>Structured for day-to-day production review</h2>
        </div>
        <div className="brand-logos" aria-label="Production workflow labels">
          {trustedBrands.map((label) => (
            <span key={label}>{label}</span>
          ))}
        </div>
      </section>

      <section className="summary-grid">
        <article className="summary-card">
          <label>Verified Cycles</label>
          <strong>{topCycleDisplay(report)}</strong>
        </article>
        <article className="summary-card">
          <label>Tracked Operators</label>
          <strong>{report.stations.length}</strong>
        </article>
        <article className="summary-card">
          <label>Present Time</label>
          <strong>{report.totals.duration_human.present}</strong>
        </article>
        <article className="summary-card">
          <label>Working Time</label>
          <strong>{report.totals.duration_human.working}</strong>
        </article>
        <article className="summary-card">
          <label>IDLE Time</label>
          <strong>{report.totals.duration_human.npt}</strong>
        </article>
        <article className="summary-card">
          <label>Line Efficiency</label>
          <strong>{report.totals.efficiency_pct}%</strong>
        </article>
      </section>

      <section className="value-ribbon">
        <article>
          <label>Overlay Review</label>
          <strong>Focused</strong>
          <span>Each processed video returns a workstation-focused overlay for quick validation.</span>
        </article>
        <article>
          <label>KPI Output</label>
          <strong>Actionable</strong>
          <span>Operator cards summarize the numbers most relevant to production supervision.</span>
        </article>
        <article>
          <label>Management View</label>
          <strong>Clear</strong>
          <span>Presence, working time, idle time, efficiency, and verified cycles appear in one board.</span>
        </article>
      </section>

      <UploadEvaluationPanel />

      <section className="stations">
        <div className="section-head">
          <div>
            <p className="eyebrow">Operator Stations</p>
            <h2>Current production performance</h2>
          </div>
          <div className="hero-chips compact">
            <span>Validated: {reliability.validated}</span>
            <span>Partial: {reliability.partial}</span>
            <span>Needs review: {reliability.unreliable}</span>
          </div>
        </div>

        <div className="station-grid">
          {report.stations.map((station) => (
            <details key={station.station_id} className="station-card station-expandable">
              <summary className="station-summary">
                <div className="station-head">
                  <div className="station-profile-head">
                    <OperatorIdentity station={station} />
                  </div>
                  <div className="station-summary-side">
                    <span className="expand-indicator" aria-hidden="true">⌄</span>
                  </div>
                </div>

                <div className="operator-inline">
                  <div>
                    <label>Presence</label>
                    <strong>{station.presence_pct}%</strong>
                  </div>
                  <div>
                    <label>Efficiency</label>
                    <strong>{station.efficiency_pct}%</strong>
                  </div>
                  <div>
                    <label>IDLE Time</label>
                    <strong>{station.duration_human.npt}</strong>
                  </div>
                </div>

                <div className="operator-kpi-row">
                  <div className={`metric${cycleDisplayClass(station)}`}>
                    <label>Verified Cycles</label>
                    <strong>{cycleDisplayCount(station)}</strong>
                  </div>
                  <div className="metric">
                    <label>Working Time</label>
                    <strong>{station.duration_human.working}</strong>
                  </div>
                  <div className={`metric${cycleDisplayClass(station)}`}>
                    <label>Average Cycle</label>
                    <strong>{cycleDisplayAverage(station)}</strong>
                  </div>
                </div>
              </summary>

              <div className="expand-panel">
                <div className="worker-profile-banner">
                  <div className="worker-profile-banner-main">
                    <div className="snapshot-meta">
                      <span className="snapshot-label">Live Snapshot</span>
                      <OperatorThumb station={station} />
                    </div>
                    <div className="operator-title-block">
                      <label>Worker Profile</label>
                      <strong>{station.demo_title}</strong>
                      <span className="operator-id-chip">ID {station.employee_id}</span>
                    </div>
                  </div>
                  <div className="worker-profile-meta">
                    <span>Shift: {station.demo_shift}</span>
                    <span>Grade: {station.demo_grade}</span>
                  </div>
                </div>
                <div className="operator-detail-grid">
                  <div className="detail-cell detail-cell-wide">
                    <UtilizationBar station={station} />
                  </div>
                  <div className="detail-cell">
                    <label>Present Time</label>
                    <strong>{station.duration_human.present}</strong>
                  </div>
                  <div className="detail-cell">
                    <label>Working Time</label>
                    <strong>{station.duration_human.working}</strong>
                  </div>
                  <div className="detail-cell">
                    <label>Cycle-Covered Work</label>
                    <strong>{station.duration_human.cycle_covered_work}</strong>
                  </div>
                  <div className="detail-cell">
                    <label>Current State</label>
                    <strong>{station.current_cycle_label || station.current_label}</strong>
                  </div>
                  <div className="detail-cell">
                    <label>Presence Loss</label>
                    <strong>{station.duration_human.absent}</strong>
                  </div>
                  <div className="detail-cell">
                    <label>Role</label>
                    <strong>{station.station_role}</strong>
                  </div>
                </div>
              </div>
            </details>
          ))}
        </div>
      </section>

      <footer className="site-footer">
        <div>
          <p className="eyebrow">Factory Dashboard</p>
          <strong>Operator performance review for industrial decision-making</strong>
        </div>
        <div className="footer-links">
          <span>Factory visibility</span>
          <span>Production analytics</span>
          <span>Workstation performance</span>
        </div>
      </footer>
    </main>
  );
}
