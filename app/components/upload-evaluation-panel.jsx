"use client";

import { useEffect, useState } from "react";

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

function stationImageUrl(report, station) {
  const params = new URLSearchParams({
    videoId: String(report?.video_id || ""),
    stationId: String(station?.station_id || ""),
  });
  return `/api/operator-image?${params.toString()}`;
}

function OperatorThumb({ report, station }) {
  const src = station?.station_image_url || stationImageUrl(report, station);
  if (src) {
    return (
      <img
        className="worker-avatar worker-avatar-image"
        src={src}
        alt={`${station.operator_name} top view`}
      />
    );
  }
  return <div className="worker-avatar">{initials(station.operator_name)}</div>;
}

function OperatorIdentity({ report, station }) {
  return (
    <div className="operator-identity">
      <div className="snapshot-meta">
        <span className="snapshot-label">Live Snapshot</span>
        <OperatorThumb report={report} station={station} />
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

function analysisStage(progress) {
  if (progress < 15) return "Preparing video";
  if (progress < 35) return "Extracting workstations";
  if (progress < 55) return "Scoring presence";
  if (progress < 80) return "Analyzing operator activity";
  if (progress < 100) return "Generating report";
  return "Complete";
}

function ReportCards({ report, overlayUrl }) {
  return (
    <section className="upload-report">
      <div className="section-head">
        <div>
          <p className="eyebrow">Generated Report</p>
          <h2>Fresh upload evaluation</h2>
        </div>
        <div className="hero-chips">
          <span>{report.video_id}</span>
          <span>Verified cycles: {topCycleDisplay(report)}</span>
          <span>Efficiency: {report.totals.efficiency_pct}%</span>
        </div>
      </div>

      <div className="overlay-tech-ribbon">
        <span>Workstation-specific overlay render</span>
        <span>Verified KPI reporting</span>
      </div>

      <div className="summary-grid">
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
          <label>Verified Cycles</label>
          <strong>{topCycleDisplay(report)}</strong>
        </article>
      </div>

      {overlayUrl ? (
        <div className="overlay-box">
          <div className="section-head">
            <div>
              <p className="eyebrow">Overlay Playback</p>
              <h2>Task-specific workstation overlay</h2>
              <p className="section-copy">
                Presence regions, tuned station activity inference, and the current best
                station-specific cycle technique are reflected in this returned video.
              </p>
            </div>
            <a className="json-link" href={overlayUrl} target="_blank" rel="noreferrer">
              Open overlay video
            </a>
          </div>
          <video className="overlay-video" src={overlayUrl} controls preload="metadata" />
        </div>
      ) : null}

      <div className="station-grid">
        {report.stations.map((station) => (
          <details key={station.station_id} className="station-card station-expandable">
            <summary className="station-summary">
              <div className="station-head">
                <div className="station-profile-head">
                  <OperatorIdentity report={report} station={station} />
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
                    <OperatorThumb report={report} station={station} />
                  </div>
                  <div className="operator-title-block">
                    <label>Worker Profile</label>
                    <strong>{station.demo_title || "Operator"}</strong>
                    <span className="operator-id-chip">ID {station.employee_id}</span>
                  </div>
                </div>
                <div className="worker-profile-meta">
                  <span>Shift: {station.demo_shift || "Day Shift"}</span>
                  <span>Grade: {station.demo_grade || "A"}</span>
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
  );
}

export default function UploadEvaluationPanel() {
  const [file, setFile] = useState(null);
  const [sampleEvery, setSampleEvery] = useState(20);
  const [presentThreshold, setPresentThreshold] = useState(0.5);
  const [device, setDevice] = useState("cuda");
  const [poseBackend, setPoseBackend] = useState("auto");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [job, setJob] = useState(null);
  const [report, setReport] = useState(null);
  const [busy, setBusy] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!job?.jobId || !["queued", "running"].includes(job?.status?.status)) {
      return;
    }
    const timer = setTimeout(async () => {
      const resp = await fetch(`/api/jobs/${job.jobId}`, { cache: "no-store" });
      const data = await resp.json();
      setJob(data);
      if (data.status?.status === "done") {
        const reportResp = await fetch(data.urls.report, { cache: "no-store" });
        const reportData = await reportResp.json();
        setReport(reportData);
        setBusy(false);
      }
      if (data.status?.status === "error") {
        setBusy(false);
      }
    }, 3000);
    return () => clearTimeout(timer);
  }, [job]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      setError("Please choose a video file first.");
      return;
    }
    setError("");
    setBusy(true);
    setReport(null);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append("video", file);
    formData.append("sampleEvery", String(sampleEvery));
    formData.append("presentThreshold", String(presentThreshold));
    formData.append("device", device);
    formData.append("poseBackend", poseBackend);
    try {
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/jobs");
        xhr.responseType = "json";
        xhr.upload.onprogress = (progressEvent) => {
          if (progressEvent.lengthComputable) {
            setUploadProgress(Math.round((progressEvent.loaded / progressEvent.total) * 100));
          }
        };
        xhr.onload = () => {
          const response = xhr.response || JSON.parse(xhr.responseText || "{}");
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(response);
          } else {
            reject(new Error(response.error || "Could not start the job."));
          }
        };
        xhr.onerror = () => reject(new Error("Upload failed. Please try again."));
        xhr.send(formData);
      });
      setUploadProgress(100);
      const statusResp = await fetch(`/api/jobs/${data.jobId}`, { cache: "no-store" });
      const statusData = await statusResp.json();
      setJob(statusData);
    } catch (uploadError) {
      setBusy(false);
      setUploadProgress(0);
      setError(uploadError.message || "Could not start the job.");
    }
  }

  return (
    <section className="upload-shell">
      <div className="section-head">
        <div>
          <p className="eyebrow">Video Analysis</p>
          <h2>Upload and process a production video</h2>
          <p className="section-copy">
            Upload a production-floor video to generate overlay playback and a station-level
            operator performance report.
          </p>
        </div>
      </div>

      <div className="upload-callouts">
        <span>Overlay-ready output</span>
        <span>Station-specific inference</span>
        <span>Executive KPI summary</span>
      </div>

      <form className="upload-form" onSubmit={handleSubmit}>
        <label className="upload-primary">
          Video file
          <input type="file" accept=".mp4,.avi,.mov,.mkv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <span className="input-help">{file ? file.name : "Select a production video file"}</span>
        </label>
        <div className="upload-actions">
          <button type="submit" disabled={busy}>
            {busy ? "Processing..." : "Start analysis"}
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={() => setShowAdvanced((value) => !value)}
          >
            {showAdvanced ? "Hide options" : "Advanced options"}
          </button>
        </div>

        {showAdvanced ? (
          <div className="advanced-grid">
            <label>
              Analysis interval
              <input type="number" min="1" value={sampleEvery} onChange={(e) => setSampleEvery(Number(e.target.value || 20))} />
            </label>
            <label>
              Presence sensitivity
              <input type="number" min="0" max="1" step="0.01" value={presentThreshold} onChange={(e) => setPresentThreshold(Number(e.target.value || 0.5))} />
            </label>
            <label>
              Processing mode
              <select value={device} onChange={(e) => setDevice(e.target.value)}>
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
            <label>
              Motion assist
              <select value={poseBackend} onChange={(e) => setPoseBackend(e.target.value)}>
                <option value="auto">auto</option>
                <option value="motion">motion</option>
                <option value="ultralytics">ultralytics</option>
              </select>
            </label>
          </div>
        ) : null}
      </form>

      {error ? <div className="error-box">{error}</div> : null}

      {busy && !job ? (
        <div className="job-box">
          <div className="job-head">
            <strong>UPLOADING</strong>
            <span>Preparing video</span>
          </div>
          <p>Your video is being prepared for processing.</p>
          <div className="progress-track">
            <div className="progress-bar" style={{ width: `${Math.max(4, uploadProgress)}%` }} />
          </div>
          <div className="job-meta">
            <span>Upload progress: {uploadProgress}%</span>
            <span>Next step: video processing</span>
          </div>
        </div>
      ) : null}

      {job ? (
        <div className="job-box">
          <div className="job-head">
            <strong>{job.status?.status?.toUpperCase()}</strong>
            <span>{analysisStage(Number(job.status?.progress || 0))}</span>
          </div>
          <p>{job.status?.message}</p>
          <div className="progress-track">
            <div className="progress-bar" style={{ width: `${Math.max(4, Number(job.status?.progress || 0))}%` }} />
          </div>
          <div className="analysis-steps" aria-hidden="true">
            {["Upload", "Presence", "Activity", "Report"].map((step, index) => {
              const thresholds = [0, 35, 70, 100];
              const active = Number(job.status?.progress || 0) >= thresholds[index];
              return (
                <div key={step} className={`analysis-step ${active ? "active" : ""}`}>
                  <span />
                  <label>{step}</label>
                </div>
              );
            })}
          </div>
          <div className="job-meta">
            <span>Progress: {job.status?.progress}%</span>
            <span>Updated: {job.status?.updated_at}</span>
            {job.urls?.report ? <a href={job.urls.report} target="_blank" rel="noreferrer">Open report JSON</a> : null}
          </div>
        </div>
      ) : null}

      {report ? <ReportCards report={report} overlayUrl={job?.urls?.overlay} /> : null}
    </section>
  );
}
