#!/usr/bin/env python3
"""Run the local dashboard upload pipeline through Stage 1, Stage 2, and report generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile-video-id", required=True)
    parser.add_argument("--roi-config", default="configs/cam33_roi_annotations.json")
    parser.add_argument("--operator-config", default="configs/altersense_operator_profiles.cam33.json")
    parser.add_argument("--stage1-checkpoint", default="artifacts/stage1/models/mobilenet_station_domain_clean/best.pt")
    parser.add_argument("--stage2-main-checkpoint", default="artifacts/stage2/models/cam33_phase_mixed_station136_hybrid_pose_v1/best.pt")
    parser.add_argument("--stage2-override-checkpoint", default="artifacts/stage2/models/cam33_phase_mixed_station1365_hybrid_pose_v1/best.pt")
    parser.add_argument("--stage2-override-stations", default="5")
    parser.add_argument("--stage2-second-override-checkpoint", default="artifacts/stage2/models/cam33_station6_phase_hybrid_pose_v2/best.pt")
    parser.add_argument("--stage2-second-override-stations", default="6")
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--clip-len", type=int, default=12)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pose-backend", choices=["auto", "ultralytics", "motion"], default="auto")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--predict-workers", type=int, default=4)
    return parser.parse_args()


def write_status(job_dir: Path, status: str, message: str, progress: int, extra: dict | None = None) -> None:
    payload = {
        "status": status,
        "message": message,
        "progress": progress,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    (job_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_step(job_dir: Path, progress: int, message: str, command: list[str]) -> None:
    write_status(job_dir, "running", message, progress, {"current_command": " ".join(command)})
    log_path = job_dir / "pipeline.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {' '.join(command)}\n")
        log_file.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def build_runtime_config(job_dir: Path, source_config: Path, upload_path: Path, profile_video_id: str, job_id: str) -> tuple[Path, str]:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    profile = next((item for item in config["videos"] if item["video_id"] == profile_video_id), None)
    if profile is None:
        raise ValueError(f"Unknown ROI profile: {profile_video_id}")
    annotated_workstations = [ws for ws in profile.get("workstations", []) if ws.get("station_roi_polygon")]
    if not annotated_workstations:
        raise ValueError(
            f"ROI profile {profile_video_id} in {source_config} has no annotated workstations. "
            "Use an annotation config with saved station polygons."
        )
    runtime_video = dict(profile)
    runtime_video_id = f"upload_{job_id}"
    runtime_video["video_id"] = runtime_video_id
    runtime_video["video_path"] = str(upload_path.resolve())
    runtime_video["source_name"] = upload_path.name
    runtime_config = {
        "version": config.get("version", 2),
        "description": "Dashboard runtime upload config",
        "videos": [runtime_video],
    }
    out_path = job_dir / "runtime_config.json"
    out_path.write_text(json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8")
    return out_path, runtime_video_id


def main() -> int:
    args = parse_args()
    job_dir = Path(args.job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    upload_path = Path(args.video).resolve()
    source_config = (REPO_ROOT / args.roi_config).resolve()
    operator_config = (REPO_ROOT / args.operator_config).resolve()
    stage1_checkpoint = (REPO_ROOT / args.stage1_checkpoint).resolve()
    stage2_main_checkpoint = (REPO_ROOT / args.stage2_main_checkpoint).resolve()
    stage2_override_checkpoint = (REPO_ROOT / args.stage2_override_checkpoint).resolve() if args.stage2_override_checkpoint else None
    stage2_second_override_checkpoint = (REPO_ROOT / args.stage2_second_override_checkpoint).resolve() if args.stage2_second_override_checkpoint else None
    override_stations = [item.strip() for item in args.stage2_override_stations.split(",") if item.strip()]
    second_override_stations = [item.strip() for item in args.stage2_second_override_stations.split(",") if item.strip()]

    if args.profile_video_id != "cam_33_28_oct_f1_24":
        raise ValueError("This dashboard upload report flow is currently locked to the cam_33 workstation layout.")

    if not upload_path.exists():
        raise FileNotFoundError(f"Uploaded video not found: {upload_path}")
    if not source_config.exists():
        raise FileNotFoundError(f"ROI config not found: {source_config}")
    if not operator_config.exists():
        raise FileNotFoundError(f"Operator config not found: {operator_config}")
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
    if not stage2_main_checkpoint.exists():
        raise FileNotFoundError(f"Stage 2 main checkpoint not found: {stage2_main_checkpoint}")
    if override_stations and stage2_override_checkpoint and not stage2_override_checkpoint.exists():
        raise FileNotFoundError(f"Stage 2 override checkpoint not found: {stage2_override_checkpoint}")
    if second_override_stations and stage2_second_override_checkpoint and not stage2_second_override_checkpoint.exists():
        raise FileNotFoundError(f"Stage 2 second override checkpoint not found: {stage2_second_override_checkpoint}")

    write_status(job_dir, "running", "Preparing runtime configuration...", 5)
    runtime_config, runtime_video_id = build_runtime_config(job_dir, source_config, upload_path, args.profile_video_id, args.job_id)

    roi_root = job_dir / "roi_crops"
    manifest_path = roi_root / "manifest.csv"
    overlay_path = job_dir / "stage1_overlay.mp4"
    presence_path = job_dir / "stage1_presence.csv"
    stage2_queue_path = job_dir / "stage2_dense_queue.csv"
    stage2_clip_path = job_dir / "stage2_dense_clips.csv"
    stage2_clip_pose_path = job_dir / "stage2_dense_clips_with_pose.csv"
    stage2_prediction_path = job_dir / "stage2_phase_predictions_main.csv"
    stage2_override_prediction_path = job_dir / "stage2_phase_predictions_override.csv"
    stage2_merged_prediction_path = job_dir / "stage2_phase_predictions_ensemble.csv"
    report_path = job_dir / "operator_report.json"

    run_step(
        job_dir,
        12,
        "Extracting workstation ROI crops from the uploaded video...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "extract_stage1_roi_crops.py"),
            "--config",
            str(runtime_config),
            "--output-dir",
            str(roi_root),
            "--sample-every",
            str(args.sample_every),
            "--video-id",
            runtime_video_id,
        ],
    )

    run_step(
        job_dir,
        24,
        "Rendering the Stage 1 workstation overlay video...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_stage1_overlay_video.py"),
            "--config",
            str(runtime_config),
            "--checkpoint",
            str(stage1_checkpoint),
            "--video-id",
            runtime_video_id,
            "--sample-every",
            str(args.sample_every),
            "--present-threshold",
            str(args.present_threshold),
            "--device",
            args.device,
            "--output",
            str(overlay_path),
        ],
    )

    run_step(
        job_dir,
        34,
        "Running Stage 1 presence scoring on ROI crops...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "predict_stage1_mobilenet.py"),
            "--checkpoint",
            str(stage1_checkpoint),
            "--manifest",
            str(manifest_path),
            "--present-threshold",
            str(args.present_threshold),
            "--device",
            args.device,
            "--output",
            str(presence_path),
        ],
    )

    run_step(
        job_dir,
        42,
        "Building dense Stage 2 frame queue for pose and motion hints...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_stage2_queue_from_manifest.py"),
            "--manifest",
            str(manifest_path),
            "--config",
            str(runtime_config),
            "--video-id",
            runtime_video_id,
            "--output",
            str(stage2_queue_path),
            "--max-per-station",
            "0",
            "--min-time-gap-sec",
            "0",
        ],
    )

    run_step(
        job_dir,
        50,
        "Generating pose and motion suggestions for the hybrid Stage 2 model...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "suggest_stage2_pose_labels.py"),
            "--queue-csv",
            str(stage2_queue_path),
            "--backend",
            args.pose_backend,
            "--pose-model",
            args.pose_model,
        ],
    )

    run_step(
        job_dir,
        58,
        "Building dense Stage 2 clips...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_stage2_clip_manifest.py"),
            "--input-csv",
            str(manifest_path),
            "--output-csv",
            str(stage2_clip_path),
            "--clip-len",
            str(args.clip_len),
            "--stride",
            str(args.clip_stride),
        ],
    )

    run_step(
        job_dir,
        64,
        "Attaching pose hints to the dense clips...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "augment_clip_manifest_with_pose.py"),
            "--clip-csv",
            str(stage2_clip_path),
            "--frame-csv",
            str(stage2_queue_path),
            "--output-csv",
            str(stage2_clip_pose_path),
        ],
    )

    run_step(
        job_dir,
        76,
        "Running the hybrid Stage 2 activity model...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "predict_stage2_clip_model.py"),
            "--checkpoint",
            str(stage2_main_checkpoint),
            "--clip-csv",
            str(stage2_clip_pose_path),
            "--batch-size",
            str(args.predict_batch_size),
            "--num-workers",
            str(args.predict_workers),
            "--device",
            args.device,
            "--output-label-column",
            "predicted_phase",
            "--output-smoothed-column",
            "smoothed_phase",
            "--output",
            str(stage2_prediction_path),
        ],
    )

    final_activity_csv = stage2_prediction_path
    if override_stations and stage2_override_checkpoint:
        run_step(
            job_dir,
            84,
            "Running the station-specific override phase model...",
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "predict_stage2_clip_model.py"),
                "--checkpoint",
                str(stage2_override_checkpoint),
                "--clip-csv",
                str(stage2_clip_pose_path),
                "--batch-size",
                str(args.predict_batch_size),
                "--num-workers",
                str(args.predict_workers),
                "--device",
                args.device,
                "--output-label-column",
                "predicted_phase",
                "--output-smoothed-column",
                "smoothed_phase",
                "--output",
                str(stage2_override_prediction_path),
            ],
        )

        run_step(
            job_dir,
            90,
            "Merging station-specific override predictions...",
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "merge_phase_prediction_overrides.py"),
                "--base-csv",
                str(stage2_prediction_path),
                "--override-csv",
                str(stage2_override_prediction_path),
                "--station-ids",
                ",".join(override_stations),
                "--output-csv",
                str(stage2_merged_prediction_path),
            ],
        )
        final_activity_csv = stage2_merged_prediction_path

    if second_override_stations and stage2_second_override_checkpoint:
        stage2_second_override_prediction_path = job_dir / "stage2_second_override_predictions.csv"
        stage2_second_merged_prediction_path = job_dir / "stage2_predictions_with_all_overrides.csv"

        run_step(
            job_dir,
            92,
            "Running the station-6 override phase model...",
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "predict_stage2_clip_model.py"),
                "--checkpoint",
                str(stage2_second_override_checkpoint),
                "--clip-csv",
                str(stage2_clip_pose_path),
                "--batch-size",
                str(args.predict_batch_size),
                "--num-workers",
                str(args.predict_workers),
                "--device",
                args.device,
                "--output-label-column",
                "predicted_phase",
                "--output-smoothed-column",
                "smoothed_phase",
                "--output",
                str(stage2_second_override_prediction_path),
            ],
        )

        run_step(
            job_dir,
            94,
            "Merging station-6 override predictions...",
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "merge_phase_prediction_overrides.py"),
                "--base-csv",
                str(final_activity_csv),
                "--override-csv",
                str(stage2_second_override_prediction_path),
                "--station-ids",
                ",".join(second_override_stations),
                "--output-csv",
                str(stage2_second_merged_prediction_path),
            ],
        )
        final_activity_csv = stage2_second_merged_prediction_path

    run_step(
        job_dir,
        96,
        "Generating the ALTERSENSE operator KPI report...",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_altersense_operator_report.py"),
            "--activity-csv",
            str(final_activity_csv),
            "--label-mode",
            "phase",
            "--activity-label-column",
            "predicted_phase",
            "--fallback-activity-label-column",
            "smoothed_label",
            "--verified-activity-label-column",
            "predicted_phase",
            "--presence-csv",
            str(presence_path),
            "--operator-config",
            str(operator_config),
            "--output",
            str(report_path),
        ],
    )

    result = {
        "job_id": args.job_id,
        "runtime_video_id": runtime_video_id,
        "profile_video_id": args.profile_video_id,
        "sample_every": args.sample_every,
        "clip_len": args.clip_len,
        "clip_stride": args.clip_stride,
        "present_threshold": args.present_threshold,
        "device": args.device,
        "pose_backend": args.pose_backend,
        "overlay_path": str(overlay_path),
        "presence_csv": str(presence_path),
        "activity_csv": str(final_activity_csv),
        "report_json": str(report_path),
        "stage2_main_checkpoint": str(stage2_main_checkpoint),
        "stage2_override_checkpoint": str(stage2_override_checkpoint) if stage2_override_checkpoint else "",
        "stage2_override_stations": override_stations,
        "stage2_second_override_checkpoint": str(stage2_second_override_checkpoint) if stage2_second_override_checkpoint else "",
        "stage2_second_override_stations": second_override_stations,
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_status(job_dir, "done", "Video evaluation complete. Report and overlay are ready.", 100, {"result_ready": True})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        job_dir_arg = None
        for idx, value in enumerate(sys.argv):
            if value == "--job-dir" and idx + 1 < len(sys.argv):
                job_dir_arg = sys.argv[idx + 1]
                break
        if job_dir_arg:
            write_status(Path(job_dir_arg), "error", f"Operator dashboard job failed:\n{exc}", 100)
        raise
