#!/usr/bin/env python3
"""Local dashcam violation pipeline.

Six idempotent, file-driven stages (see SOLUTION.md §4). Each stage reads and
writes per-clip manifest.json files; re-running any stage is safe.

    ingest  -> hash + dedup + copy + organize, write initial manifest
    detect  -> cheap YOLO gate: candidate vs cleared
    review  -> VLM judgment (vLLM by default): structured verdict -> pending_review
    list    -> show the pending-review queue
    run     -> ingest + detect + review in sequence

Runs natively on the host or inside the worker container. YOLO/torch are
imported lazily, so the VLM-only path (use_yolo_prefilter: false) needs neither.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
import yaml

# --- status state machine (SOLUTION.md §5) ---------------------------------
#   ingested -> candidate ----> pending_review ----> approved
#           \-> cleared     \-> cleared          \-> rejected
ST_INGESTED = "ingested"
ST_CANDIDATE = "candidate"
ST_CLEARED = "cleared"
ST_PENDING = "pending_review"
ST_ERROR = "error"        # video could not be decoded — surfaced, not silently cleared

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config & small helpers
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    # Env overrides for the VLM endpoint/model so the serving backend (Ollama or
    # vLLM — both OpenAI-compatible) is swappable without editing config.yaml.
    # VLM_BASE_URL is preferred; OLLAMA_BASE_URL kept for back-compat.
    env_url = os.environ.get("VLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
    if env_url:
        cfg.setdefault("vlm", {})["base_url"] = env_url
    env_model = os.environ.get("VLM_MODEL")
    if env_model:
        cfg.setdefault("vlm", {})["model"] = env_model
    return cfg


def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def data_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"])


def clips_root(cfg: dict) -> Path:
    return data_dir(cfg) / "clips"


def ledger_path(cfg: dict) -> Path:
    return data_dir(cfg) / "ingested_hashes.txt"


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    """Atomic write so a crash mid-write never corrupts state."""
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, manifest_path)


def iter_manifests(cfg: dict) -> Iterable[tuple[Path, dict]]:
    root = clips_root(cfg)
    if not root.exists():
        return
    for mpath in sorted(root.glob("*/*/manifest.json")):
        try:
            with open(mpath) as fh:
                yield mpath, json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log(f"WARN  skipping unreadable manifest {mpath}: {exc}")


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe frame extraction (no opencv dependency for the VLM path)
# ---------------------------------------------------------------------------
def ffprobe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def parse_name_date(name: str) -> Optional[str]:
    """Dashcams encode capture time in the filename (e.g. REC2_20260620_163913).
    This is more reliable than mtime (which copying resets) so it is tried first.
    Matches an 8-digit YYYYMMDD anywhere in the name."""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if not m:
        return None
    y, mo, d = m.groups()
    try:
        return dt.date(int(y), int(mo), int(d)).isoformat()
    except ValueError:
        return None


def ffprobe_creation_date(video: Path) -> Optional[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True,
    )
    stamp = out.stdout.strip()
    if stamp:
        try:
            return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return None
    return None


def _jpeg_qscale(quality: int) -> int:
    # Map 0-100 (higher=better) to ffmpeg mjpeg -q:v 2-31 (lower=better).
    quality = max(1, min(100, quality))
    return max(2, round(2 + (100 - quality) / 100 * 29))


def extract_frame_jpeg(video: Path, ts: float, max_width: int, quality: int) -> Optional[bytes]:
    """Decode a single frame at `ts` seconds as JPEG bytes (downscaled)."""
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{ts:.3f}", "-i", str(video),
        "-frames:v", "1",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", str(_jpeg_qscale(quality)), "-",
    ]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or not out.stdout:
        return None
    return out.stdout


def sample_timestamps(duration: float, n: int) -> list[float]:
    """N timestamps spread across the clip, biased away from the very ends."""
    if duration <= 0:
        return [0.0] * n
    return [round(duration * (i + 0.5) / n, 2) for i in range(n)]


def extract_gate_frames(video: Path, sample_fps: float, out_dir: Path) -> list[Path]:
    """Extract frames at `sample_fps` for the YOLO gate; returns file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "f_%05d.jpg")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"fps={sample_fps}", "-q:v", "4", pattern],
        capture_output=True,
    )
    return sorted(out_dir.glob("f_*.jpg"))


# ---------------------------------------------------------------------------
# Stage 1: ingest  (scan SD -> hash -> dedup -> copy -> organize -> manifest)
# ---------------------------------------------------------------------------
def hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ledger(cfg: dict) -> set[str]:
    p = ledger_path(cfg)
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def append_ledger(cfg: dict, digest: str) -> None:
    p = ledger_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(digest + "\n")
        fh.flush()
        os.fsync(fh.fileno())  # durable: a half-written final line breaks dedup


def iter_source_videos(sd_mount: Path, exts: list[str]) -> Iterable[Path]:
    exts = {e.lower() for e in exts}
    for p in sorted(sd_mount.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def ingest(cfg: dict) -> None:
    sd = Path(cfg["paths"]["sd_mount"])
    if not sd.exists():
        log(f"ingest: SD mount {sd} not found — nothing to ingest")
        return
    icfg = cfg["ingest"]
    ledger = load_ledger(cfg)
    new_count = skip_count = 0

    for src in iter_source_videos(sd, icfg["video_extensions"]):
        part = None
        try:
            digest = hash_file(src, icfg["hash_algorithm"])
            if digest in ledger:
                skip_count += 1
                continue

            clip_id = digest[: icfg["clip_id_length"]]
            captured = (
                parse_name_date(src.name)
                or ffprobe_creation_date(src)
                or dt.date.fromtimestamp(src.stat().st_mtime).isoformat()
            )
            clip_dir = clips_root(cfg) / captured / clip_id
            # Preserve the source container/extension byte-for-byte (evidence
            # integrity): an .avi stays original.avi, never renamed to .mp4.
            video_name = "original" + src.suffix.lower()
            dest = clip_dir / video_name
            manifest_path = clip_dir / "manifest.json"

            # Self-heal: an existing manifest means this clip was already
            # ingested (e.g. a crash lost the ledger entry). Re-record the hash
            # and skip — never re-copy over the read-only original.
            if manifest_path.exists():
                append_ledger(cfg, digest)
                ledger.add(digest)
                skip_count += 1
                continue

            clip_dir.mkdir(parents=True, exist_ok=True)
            # Atomic ingest: copy to a temp .part, then os.replace() into place.
            # rename() replaces the destination atomically and — unlike copy —
            # succeeds even if an old read-only original is already there
            # (it needs dir write, not file write). A crash mid-copy leaves only
            # a .part, never a half-written "original" and never a no-op-breaking
            # PermissionError on the next run.
            part = clip_dir / (video_name + ".part")
            shutil.copy2(src, part)
            os.replace(part, dest)
            part = None
            if icfg.get("set_readonly", True):
                try:
                    os.chmod(dest, 0o444)
                except OSError:
                    # Network shares (CIFS/NFS) often pin file modes; the
                    # pipeline never writes the original anyway, so don't let a
                    # rejected chmod abort the clip's ingest.
                    pass

            manifest = {
                "clip_id": clip_id,
                "sha1": digest,
                "source_name": src.name,
                "captured_date": captured,
                "ingested_at": now_iso(),
                "video": video_name,
                "duration_s": round(ffprobe_duration(dest), 2),
                "candidate": None,
                "yolo_hits": None,
                "violations": [],
                "vlm": None,
                "reviewed_at": None,
                "status": ST_INGESTED,
            }
            save_manifest(manifest_path, manifest)
            append_ledger(cfg, digest)
            ledger.add(digest)
            new_count += 1
            log(f"ingest: + {src.name} -> {captured}/{clip_id}")
        except Exception as exc:
            # One bad clip must not abort the whole scan.
            log(f"ingest: ERROR on {src.name}: {exc} — skipping")
            if part is not None:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass

    log(f"ingest: {new_count} new, {skip_count} duplicates skipped")


# ---------------------------------------------------------------------------
# Stage 3: detect  (cheap YOLO gate)
# ---------------------------------------------------------------------------
_YOLO_MODEL = None  # lazily loaded singleton


def get_yolo(cfg: dict):
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO  # heavy import, only when gating
        log(f"detect: loading YOLO model {cfg['detect']['model']} ...")
        _YOLO_MODEL = YOLO(cfg["detect"]["model"])
    return _YOLO_MODEL


def detect(cfg: dict, force: bool = False) -> None:
    dcfg = cfg["detect"]
    targets = {ST_INGESTED} | ({ST_CANDIDATE, ST_CLEARED} if force else set())

    if not dcfg.get("use_yolo_prefilter", True):
        n = 0
        for mpath, m in iter_manifests(cfg):
            if m["status"] not in targets:
                continue
            m.update(candidate=True, yolo_hits=None, status=ST_CANDIDATE)
            save_manifest(mpath, m)
            n += 1
        log(f"detect: gate disabled -> {n} clips marked candidate")
        return

    triggers = set(dcfg["trigger_classes"])
    device = dcfg.get("device") or None
    processed = candidates = errors = 0

    for mpath, m in iter_manifests(cfg):
        if m["status"] not in targets:
            continue
        video = mpath.parent / m["video"]
        with tempfile.TemporaryDirectory() as td:
            frames = extract_gate_frames(video, dcfg["sample_fps"], Path(td))
            if not frames:
                # No decodable frames -> the clip is unreadable/corrupt. Flag it
                # for attention instead of silently clearing it (which would
                # drop the evidence from the queue forever).
                m["status"] = ST_ERROR
                save_manifest(mpath, m)
                errors += 1
                log(f"detect: {m['clip_id']} no frames decoded -> {ST_ERROR}")
                continue
            hits = 0
            model = get_yolo(cfg)
            results = model(
                [str(f) for f in frames], imgsz=dcfg["imgsz"],
                conf=dcfg["conf"], device=device, verbose=False,
            )
            names = model.names
            for r in results:
                for c in r.boxes.cls.tolist():
                    if names[int(c)] in triggers:
                        hits += 1

        is_candidate = hits >= dcfg["min_hits"]
        m["yolo_hits"] = hits
        m["candidate"] = is_candidate
        m["status"] = ST_CANDIDATE if is_candidate else ST_CLEARED
        save_manifest(mpath, m)
        processed += 1
        candidates += int(is_candidate)
        log(f"detect: {m['clip_id']} hits={hits} -> {m['status']}")

    msg = f"detect: {processed} gated, {candidates} candidates"
    if errors:
        msg += f", {errors} unreadable -> {ST_ERROR}"
    log(msg)


# ---------------------------------------------------------------------------
# Stage 4: review  (VLM judgment via an OpenAI-compatible endpoint; vLLM default)
# ---------------------------------------------------------------------------
def build_prompt(cfg: dict, frame_times: list[float]) -> str:
    rules = cfg["rules"]
    rule_lines = "\n".join(
        f'  - "{r["id"]}": {" ".join(r["text"].split())}' for r in rules
    )
    times = ", ".join(f"frame {i+1} = {t:.1f}s" for i, t in enumerate(frame_times))
    return (
        "You are a traffic-safety reviewer examining still frames sampled from a "
        "single dashcam clip. The camera is mounted in the EGO car (the car the "
        "camera is in). Review ONLY the behaviour of OTHER road users — never "
        "report the ego car itself.\n\n"
        f"The frames, in order, correspond to these clip timestamps: {times}.\n\n"
        "Report a violation ONLY when one of these rules is clearly visible. Do "
        "not speculate; if you cannot see it, do not report it:\n"
        f"{rule_lines}\n\n"
        "For each violation, transcribe the offending vehicle's licence plate if "
        "it is legible; otherwise use an empty string. Estimate frame_time as the "
        "clip timestamp (seconds) of the frame that best shows the violation. "
        "confidence is your certainty from 0.0 to 1.0.\n\n"
        "Respond with STRICT JSON ONLY, no prose, no markdown fences, exactly:\n"
        '{"violations":[{"rule_id":"","confidence":0.0,"frame_time":0.0,'
        '"plate":"","description":""}]}\n'
        "If you see no violation by other road users, return "
        '{"violations":[]}.'
    )


def parse_vlm_json(content: str) -> Optional[dict]:
    """Tolerate code fences and leading/trailing prose; return None on failure."""
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Decode just the first JSON object starting at the first '{', ignoring any
    # trailing prose/objects. (A greedy `\{.*\}` would swallow trailing junk and
    # fail on otherwise-recoverable output.)
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def call_vlm(cfg: dict, prompt: str, frames_b64: list[str]) -> tuple[Optional[dict], str]:
    vc = cfg["vlm"]
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    payload = {
        "model": vc["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": vc.get("temperature", 0.1),
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    url = vc["base_url"].rstrip("/") + "/v1/chat/completions"
    resp = requests.post(url, json=payload, timeout=vc.get("timeout_seconds", 300))
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return parse_vlm_json(raw), raw


def review(cfg: dict, force: bool = False) -> None:
    vc = cfg["vlm"]
    targets = {ST_CANDIDATE} | ({ST_PENDING, ST_CLEARED} if force else set())
    reviewed = flagged = 0

    for mpath, m in iter_manifests(cfg):
        if m["status"] not in targets:
            continue
        # Never re-review something a human already actioned.
        if m["status"] in ("approved", "rejected"):
            continue
        if m["status"] == ST_CLEARED and not m.get("candidate"):
            continue

        video = mpath.parent / m["video"]
        duration = m.get("duration_s") or ffprobe_duration(video)
        times = sample_timestamps(duration, vc["frames_per_window"])
        # Keep timestamps in lockstep with the frames that actually decoded, so
        # the per-frame timestamps in the prompt line up 1:1 with the images. If
        # a middle frame fails to extract, dropping its timestamp too prevents
        # mislabelling every later frame (which would misdirect the human-review
        # jump-to-timestamp UX).
        kept_times: list[float] = []
        frames_b64: list[str] = []
        for ts in times:
            jpeg = extract_frame_jpeg(video, ts, vc["max_frame_width"], vc["jpeg_quality"])
            if jpeg:
                kept_times.append(ts)
                frames_b64.append(base64.b64encode(jpeg).decode("ascii"))
        if not frames_b64:
            log(f"review: {m['clip_id']} no frames extracted — skipping")
            continue

        prompt = build_prompt(cfg, kept_times)
        try:
            parsed, raw = call_vlm(cfg, prompt, frames_b64)
        except requests.RequestException as exc:
            log(f"review: {m['clip_id']} VLM call failed: {exc}")
            continue

        violations = []
        if isinstance(parsed, dict) and isinstance(parsed.get("violations"), list):
            for v in parsed["violations"]:
                if not isinstance(v, dict):
                    continue
                try:
                    conf = float(v.get("confidence", 0))
                except (TypeError, ValueError):
                    conf = 0.0
                try:
                    frame_time = float(v.get("frame_time", 0) or 0)
                except (TypeError, ValueError):
                    frame_time = 0.0  # VLM sometimes emits "3.5s"/"unknown"
                if conf >= vc["min_confidence"]:
                    violations.append({
                        "rule_id": str(v.get("rule_id", "")),
                        "confidence": round(conf, 3),
                        "frame_time": round(max(0.0, frame_time), 2),
                        "plate": str(v.get("plate", "")),
                        "description": str(v.get("description", "")),
                    })

        m["violations"] = violations
        m["vlm"] = {"raw": raw, "parsed_ok": parsed is not None,
                    "frames": len(frames_b64), "model": vc["model"]}
        m["reviewed_at"] = now_iso()
        m["status"] = ST_PENDING if violations else ST_CLEARED
        save_manifest(mpath, m)
        reviewed += 1
        flagged += int(bool(violations))
        log(f"review: {m['clip_id']} -> {m['status']} "
            f"({len(violations)} violation(s))")

    log(f"review: {reviewed} reviewed, {flagged} -> pending_review")


# ---------------------------------------------------------------------------
# Stage 5: list the pending-review queue
# ---------------------------------------------------------------------------
def list_pending(cfg: dict) -> None:
    rows = 0
    for mpath, m in iter_manifests(cfg):
        if m["status"] != ST_PENDING:
            continue
        rows += 1
        print(f"\n{m['captured_date']}/{m['clip_id']}  ({m['source_name']})")
        print(f"  {mpath.parent / m['video']}")
        for v in m["violations"]:
            plate = f" plate={v['plate']}" if v.get("plate") else ""
            print(f"  - [{v['rule_id']}] conf={v['confidence']} "
                  f"@ {v['frame_time']}s{plate}: {v['description']}")
    if rows == 0:
        print("Queue empty — no clips pending review.")
    else:
        print(f"\n{rows} clip(s) pending review.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    sd = args.sd or os.environ.get("SD_MOUNT")
    data = args.data or os.environ.get("DATA_DIR")
    if sd:
        cfg["paths"]["sd_mount"] = sd
    if data:
        cfg["paths"]["data_dir"] = data


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Local dashcam violation pipeline")
    p.add_argument("stage", choices=["ingest", "detect", "review", "list", "run"],
                   help="pipeline stage to run ('run' = ingest+detect+review)")
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--sd", help="override SD mount path")
    p.add_argument("--data", help="override data dir path")
    p.add_argument("--force", action="store_true",
                   help="re-run detect/review over already-processed clips")
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config))
    apply_overrides(cfg, args)

    if args.stage in ("ingest", "run"):
        ingest(cfg)
    if args.stage in ("detect", "run"):
        detect(cfg, force=args.force)
    if args.stage in ("review", "run"):
        review(cfg, force=args.force)
    if args.stage == "list":
        list_pending(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
