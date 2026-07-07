import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import sampleReport from "@/app/data/sample_operator_report.json";

const DEFAULT_REPORT_PATH =
  "app/data/sample_operator_report.json";

const STATION_REPORT_OVERRIDES = {
  "4": "artifacts/altersense/operator_report_cam33_station4_override_v3.json",
  "6": "artifacts/altersense/operator_report_cam33_station6_phase_override_v3.json",
};

function formatSeconds(totalSeconds) {
  const total = Math.max(0, Math.round(Number(totalSeconds || 0)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  }
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function normalizeOverrideStation(station) {
  const verified =
    station?.verified_cycle_count ??
    station?.cycle_count ??
    0;
  const averageCycle =
    station?.average_cycle_time_sec ??
    0;
  return {
    ...station,
    verified_cycle_count: verified,
    cycle_count: station?.cycle_count ?? verified,
    cycle_kpi_display: station?.cycle_kpi_display ?? {
      count: verified > 0 ? verified : null,
      avg_cycle: averageCycle > 0 ? formatSeconds(averageCycle) : "Low confidence",
      status: verified > 0 ? "partial" : "low_confidence",
    },
  };
}

function computeOperatorGrade(station) {
  const efficiency = Number(station?.efficiency_pct ?? 0);
  const presence = Number(station?.presence_pct ?? 0);
  const cycleStatus = String(station?.cycle_kpi?.status || "low_confidence");
  const verifiedCycles = Number(station?.verified_cycle_count ?? station?.cycle_count ?? 0);

  let score = 0;
  score += Math.max(0, Math.min(100, efficiency)) * 0.65;
  score += Math.max(0, Math.min(100, presence)) * 0.2;

  if (cycleStatus === "ok") {
    score += 15;
  } else if (cycleStatus === "partial") {
    score += 8;
  } else if (verifiedCycles > 0) {
    score += 3;
  }

  if (cycleStatus === "low_confidence") {
    score -= 10;
  }
  if (verifiedCycles <= 0) {
    score -= 5;
  }

  if (score >= 92) return "A";
  if (score >= 86) return "A-";
  if (score >= 80) return "B+";
  if (score >= 72) return "B";
  if (score >= 64) return "C+";
  if (score >= 56) return "C";
  return "D";
}

const DEMO_WORKERS = {
  "1": {
    operator_name: "Rahima Akter",
    employee_id: "AS-2401",
    operation_name: "Armhole Topstitch",
    demo_title: "Senior Sewing Operator",
    demo_shift: "Day Shift",
    demo_grade: "A",
  },
  "2": {
    operator_name: "Md. Rakib Hossain",
    employee_id: "AS-2402",
    operation_name: "Shoulder Join",
    demo_title: "Sewing Operator",
    demo_shift: "Day Shift",
    demo_grade: "A-",
  },
  "3": {
    operator_name: "Nasima Begum",
    employee_id: "AS-2403",
    operation_name: "Sleeve Attach",
    demo_title: "Line Operator",
    demo_shift: "Day Shift",
    demo_grade: "B+",
  },
  "4": {
    operator_name: "Rubi Khatun",
    employee_id: "AS-2404",
    operation_name: "Collar Attach",
    demo_title: "Special Machine Operator",
    demo_shift: "Day Shift",
    demo_grade: "A",
  },
  "5": {
    operator_name: "Mst. Jannat Ara",
    employee_id: "AS-2405",
    operation_name: "Bottom Hem",
    demo_title: "Sewing Operator",
    demo_shift: "Day Shift",
    demo_grade: "B+",
  },
  "6": {
    operator_name: "Md. Sohel Rana",
    employee_id: "AS-2406",
    operation_name: "Neck Rib Attach",
    demo_title: "Special Machine Operator",
    demo_shift: "Day Shift",
    demo_grade: "A-",
  },
  "7": {
    operator_name: "Farzana Akter",
    employee_id: "AS-2407",
    operation_name: "Preparation / Pass Station",
    demo_title: "Preparation Operator",
    demo_shift: "Day Shift",
    demo_grade: "A",
  },
};

export function resolveReportPath() {
  const configured = process.env.REPORT_JSON_PATH || DEFAULT_REPORT_PATH;
  return path.isAbsolute(configured) ? configured : path.join(process.cwd(), configured);
}

function emptyReport() {
  return {
    video_id: "pending_upload",
    stations: [],
    totals: {
      verified_cycle_count: 0,
      display_verified_cycle_count: 0,
      efficiency_pct: 0,
      cycle_kpi: { status: "low_confidence" },
      duration_human: {
        present: "0m 00s",
        working: "0m 00s",
        npt: "0m 00s",
      },
    },
    meta: {
      source: "fallback",
      message: "No packaged report found. Upload a video or connect the GPU worker to generate a live report.",
    },
  };
}

function recomputeTotals(report, stations) {
  const presentDurationSec = stations.reduce(
    (sum, station) => sum + Number(station.present_duration_sec || 0),
    0
  );
  const workingDurationSec = stations.reduce(
    (sum, station) =>
      sum +
      Number(station.productive_duration_sec || 0) +
      Number(station.support_work_duration_sec || 0),
    0
  );
  const idleDurationSec = stations.reduce((sum, station) => {
    if (String(station.station_role || "") === "prep_pass") {
      return sum;
    }
    return sum + Number(station.npt_duration_sec || 0);
  }, 0);
  const trackedOperators = stations.length;
  const efficiencyPct = presentDurationSec
    ? Number(((workingDurationSec / presentDurationSec) * 100).toFixed(2))
    : 0;

  return {
    ...(report.totals || {}),
    stations: trackedOperators,
    present_duration_sec: presentDurationSec,
    working_duration_sec: workingDurationSec,
    npt_duration_sec: idleDurationSec,
    efficiency_pct: efficiencyPct,
    duration_human: {
      ...(report.totals?.duration_human || {}),
      present: formatSeconds(presentDurationSec),
      working: formatSeconds(workingDurationSec),
      npt: formatSeconds(idleDurationSec),
    },
  };
}

function normalizeVideoId(videoId) {
  return String(videoId || "").trim();
}

function stationImageCandidates(videoId, stationId) {
  const cleanVideoId = normalizeVideoId(videoId);
  const cleanStationId = String(stationId || "").trim();
  if (!cleanVideoId || !cleanStationId) {
    return [];
  }
  return [
    path.join(
      process.cwd(),
      "public",
      "operator_snapshots",
      cleanVideoId === "cam_33_28_oct_f1_24" ? "cam33" : cleanVideoId,
      `station_${cleanStationId}_masked_crop.jpg`
    ),
    path.join(
      process.cwd(),
      "artifacts",
      "stage1",
      "visualizations",
      "cam33_roi_previews",
      cleanVideoId,
      `station_${cleanStationId}_masked_crop.jpg`
    ),
    path.join(
      process.cwd(),
      "artifacts",
      "stage1",
      "visualizations",
      "roi_previews",
      cleanVideoId,
      `station_${cleanStationId}_masked_crop.jpg`
    ),
    path.join(
      process.cwd(),
      "datasets",
      "processed",
      "stage1",
      "domain_reference_crops",
      "present",
      `${cleanVideoId}_station_${cleanStationId}_frame_000000.jpg`
    ),
  ];
}

export async function resolveStationImagePath(videoId, stationId) {
  for (const candidate of stationImageCandidates(videoId, stationId)) {
    try {
      await stat(candidate);
      return candidate;
    } catch {
      continue;
    }
  }
  return null;
}

export function stationImageUrl(videoId, stationId) {
  const params = new URLSearchParams({
    videoId: normalizeVideoId(videoId),
    stationId: String(stationId || "").trim(),
  });
  return `/api/operator-image?${params.toString()}`;
}

export function publicStationImageUrl(videoId, stationId) {
  const cleanVideoId = normalizeVideoId(videoId);
  const bucket = cleanVideoId === "cam_33_28_oct_f1_24" ? "cam33" : cleanVideoId;
  return `/operator_snapshots/${bucket}/station_${String(stationId || "").trim()}_masked_crop.jpg`;
}

export async function loadReport() {
  const reportPath = resolveReportPath();
  let report;
  try {
    if ((process.env.REPORT_JSON_PATH || DEFAULT_REPORT_PATH) === DEFAULT_REPORT_PATH) {
      report = sampleReport;
    } else {
      const raw = await readFile(reportPath, "utf-8");
      report = JSON.parse(raw);
    }
  } catch (error) {
    if (error?.code === "ENOENT") {
      report = sampleReport || emptyReport();
    } else {
      throw error;
    }
  }
  const overrideStations = {};

  for (const [stationId, overridePath] of Object.entries(STATION_REPORT_OVERRIDES)) {
    try {
      const overrideRaw = await readFile(path.join(process.cwd(), overridePath), "utf-8");
      const overrideReport = JSON.parse(overrideRaw);
      const overrideStation = (overrideReport.stations || []).find(
        (station) => String(station.station_id) === stationId
      );
      if (overrideStation) {
        overrideStations[stationId] = {
          ...normalizeOverrideStation(overrideStation),
          report_override_source: path.basename(overridePath),
        };
      }
    } catch {
      continue;
    }
  }

  const stations = (report.stations || []).map((station) => {
    const stationId = String(station.station_id);
    const mergedStation = overrideStations[stationId]
      ? {
          ...station,
          ...overrideStations[stationId],
          station_id: station.station_id,
        }
      : station;
    const demo = DEMO_WORKERS[String(station.station_id)] || {};
    return {
      ...mergedStation,
      operator_name: demo.operator_name || mergedStation.operator_name,
      employee_id: demo.employee_id || mergedStation.employee_id,
      operation_name: demo.operation_name || mergedStation.operation_name,
      demo_title: demo.demo_title || mergedStation.demo_title || "Operator",
      demo_shift: demo.demo_shift || mergedStation.demo_shift || "Day Shift",
      demo_grade: computeOperatorGrade(mergedStation),
      station_image_url:
        normalizeVideoId(report.video_id) === "cam_33_28_oct_f1_24"
          ? publicStationImageUrl(report.video_id, station.station_id)
          : stationImageUrl(report.video_id, station.station_id),
    };
  });
  return {
    ...report,
    stations,
    totals: recomputeTotals(report, stations),
  };
}

export function summarizeReliability(stations) {
  const counts = { validated: 0, partial: 0, unreliable: 0 };
  for (const station of stations) {
    const badge = station.reliability_badge || "partial";
    counts[badge] = (counts[badge] || 0) + 1;
  }
  return counts;
}

export function pipelineDefinition() {
  return {
    title: "Production Pipeline",
    stages: [
      "Stage 1: fixed workstation ROI + MobileNetV3 presence gating",
      "Stage 2: hybrid clip + pose activity prediction",
      "Cycle reporting: verified cycles only for KPI trust, heuristic cycles for audit only"
    ],
    deployment:
      "Vercel should host the dashboard UI. Heavy video inference should run on a separate GPU worker or on-prem service."
  };
}
