#!/usr/bin/env python3
"""Human review UI for the dashcam violation queue (SOLUTION.md §9, §11 task 1).

Closes the human-approval loop: list every clip with status `pending_review`,
play it in mpv jumped to the flagged moment, and write `approved`/`rejected`
back to its manifest.json.

This is a HOST tool, not a container one — it drives the host's mpv over its
JSON IPC socket and shows a real window. It is intentionally stdlib-only so it
runs with the system `python3` (no venv, no pip):

    python3 review_ui.py --data ./data            # review the queue in mpv
    python3 review_ui.py --data ./data --no-mpv   # decide from stills/CLI only

State machine (matches worker/pipeline.py):
    pending_review --approve--> approved
                   --reject---> rejected
A `human_review` block (decision, reviewer, timestamp, note) is added for audit.
The original clip is never modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ST_PENDING = "pending_review"
ST_APPROVED = "approved"
ST_REJECTED = "rejected"

PREROLL_S = 3.0  # start playback this many seconds before the flagged moment


# ---------------------------------------------------------------------------
# Manifest helpers (kept tiny so this tool needs no third-party deps)
# ---------------------------------------------------------------------------
def find_pending(data_dir: Path) -> list[tuple[Path, dict]]:
    root = data_dir / "clips"
    out: list[tuple[Path, dict]] = []
    if not root.exists():
        return out
    for mpath in sorted(root.glob("*/*/manifest.json")):
        try:
            with open(mpath) as fh:
                m = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("status") == ST_PENDING:
            out.append((mpath, m))
    return out


def save_manifest(mpath: Path, manifest: dict) -> None:
    tmp = mpath.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, mpath)


def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# mpv JSON IPC controller — one persistent window we seek around in
# ---------------------------------------------------------------------------
class Mpv:
    def __init__(self) -> None:
        self.sock_path = os.path.join(
            tempfile.gettempdir(), f"dv-mpv-{os.getpid()}.sock"
        )
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self.alive = False
        self._buf = b""
        self._rid = 0

    def start(self) -> None:
        """Initial launch. Raises if mpv isn't installed (so the caller can
        fall back to --no-mpv mode); otherwise spawns and connects."""
        if shutil.which("mpv") is None:
            raise RuntimeError("mpv not found on PATH")
        self._spawn()

    def _spawn(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        if os.path.exists(self.sock_path):
            try:
                os.unlink(self.sock_path)
            except OSError:
                pass
        self._buf = b""
        self.proc = subprocess.Popen(
            ["mpv", "--idle=yes", "--force-window=yes", "--keep-open=yes",
             "--no-terminal", "--osd-level=1", "--osd-duration=4000",
             "--title=Dashcam review", f"--input-ipc-server={self.sock_path}"],
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            if os.path.exists(self.sock_path):
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.sock_path)
                    self.sock = s
                    self.alive = True
                    return
                except OSError:
                    pass
            if self.proc.poll() is not None:
                raise RuntimeError("mpv exited before IPC was ready")
            time.sleep(0.1)
        raise RuntimeError("timed out waiting for mpv IPC socket")

    def ensure_started(self) -> bool:
        """Relaunch mpv if it died (e.g. the user closed the window). Returns
        False only if mpv genuinely can't be (re)started."""
        if self.proc is not None and self.proc.poll() is None and self.alive:
            return True
        try:
            self._spawn()
            return True
        except RuntimeError:
            self.alive = False
            return False

    def _send(self, command: list) -> dict:
        """Send a command; never raises — returns {"error": ...} on a dead pipe
        so a closed mpv window degrades gracefully instead of crashing."""
        if not self.alive or self.sock is None:
            return {"error": "mpv-not-running"}
        self._rid += 1
        rid = self._rid
        try:
            self.sock.sendall(
                (json.dumps({"command": command, "request_id": rid}) + "\n").encode())
        except OSError:
            self.alive = False
            return {"error": "send-failed"}
        deadline = time.time() + 10
        while time.time() < deadline:
            while b"\n" not in self._buf:
                try:
                    chunk = self.sock.recv(65536)
                except OSError:
                    self.alive = False
                    return {"error": "recv-failed"}
                if not chunk:
                    self.alive = False
                    return {"error": "disconnected"}
                self._buf += chunk
            line, self._buf = self._buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("request_id") == rid:
                return msg
        return {"error": "timeout"}

    def play_at(self, video: Path, start: float) -> bool:
        """Load `video` and start playback `start` seconds in. Returns False if
        mpv is unavailable (window closed and couldn't be relaunched)."""
        if not self.ensure_started():
            return False
        if self._send(["loadfile", str(video), "replace"]).get("error"):
            return False
        # Wait until the file is loaded (duration becomes available), then seek.
        deadline = time.time() + 10
        while time.time() < deadline:
            r = self._send(["get_property", "duration"])
            if isinstance(r.get("data"), (int, float)):
                break
            if r.get("error"):
                return False
            time.sleep(0.1)
        self._send(["seek", max(0.0, start), "absolute", "exact"])
        self._send(["set_property", "pause", False])
        return self.alive

    def osd(self, text: str) -> None:
        if self.alive:
            self._send(["show-text", text, 4000])

    def stop(self) -> None:
        try:
            if self.sock is not None and self.alive:
                self._send(["quit"])
                self.sock.close()
        except OSError:
            pass
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()
        if os.path.exists(self.sock_path):
            try:
                os.unlink(self.sock_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Review loop
# ---------------------------------------------------------------------------
def describe(m: dict, idx: int, total: int) -> None:
    print(f"\n{'='*72}")
    print(f"[{idx}/{total}]  {m.get('captured_date')}/{m.get('clip_id')}  "
          f"({m.get('source_name')})")
    for i, v in enumerate(m.get("violations", []), 1):
        plate = f"  plate={v['plate']}" if v.get("plate") else ""
        print(f"  {i}. [{v.get('rule_id')}] conf={v.get('confidence')} "
              f"@ {v.get('frame_time')}s{plate}")
        print(f"     {v.get('description', '')}")
    print(f"{'='*72}")


def record_decision(mpath: Path, m: dict, decision: str, note: str) -> None:
    m["status"] = ST_APPROVED if decision == "approve" else ST_REJECTED
    m["human_review"] = {
        "decision": decision,
        "reviewer": getpass.getuser(),
        "reviewed_at": now_iso(),
        "note": note,
    }
    save_manifest(mpath, m)


PROMPT = ("  [a]pprove  [r]eject  [s]kip  [p]lay again  "
          "[n]ext violation  [q]uit > ")


def review_queue(data_dir: Path, use_mpv: bool) -> int:
    pending = find_pending(data_dir)
    if not pending:
        print("Queue empty — no clips pending review.")
        return 0

    print(f"{len(pending)} clip(s) pending review.")
    mpv = None
    if use_mpv:
        try:
            mpv = Mpv()
            mpv.start()
        except RuntimeError as exc:
            print(f"(mpv unavailable: {exc} — falling back to --no-mpv mode)")
            mpv = None

    approved = rejected = skipped = 0
    try:
        for idx, (mpath, m) in enumerate(pending, 1):
            video = mpath.parent / m.get("video", "")
            violations = m.get("violations", []) or [{}]
            vi = 0

            def cue(i: int) -> None:
                ft = float(violations[i].get("frame_time", 0) or 0)
                played = False
                if mpv is not None and video.exists():
                    played = mpv.play_at(video, ft - PREROLL_S)
                    if played:
                        mpv.osd(f"{m.get('clip_id')}  "
                                f"{violations[i].get('rule_id','')} @ {ft:.1f}s")
                if not played:
                    # mpv missing, window closed, or file unreadable — let the
                    # reviewer still see the moment by hand without crashing.
                    print(f"  ▶ play manually:  mpv --start={ft:.1f} '{video}'")

            describe(m, idx, len(pending))
            cue(0)

            while True:
                try:
                    choice = input(PROMPT).strip().lower()
                except EOFError:
                    choice = "q"
                if choice in ("a", "approve"):
                    note = input("  note (optional): ").strip()
                    record_decision(mpath, m, "approve", note)
                    approved += 1
                    print("  -> approved")
                    break
                if choice in ("r", "reject"):
                    note = input("  note (optional): ").strip()
                    record_decision(mpath, m, "reject", note)
                    rejected += 1
                    print("  -> rejected")
                    break
                if choice in ("s", "skip", ""):
                    skipped += 1
                    print("  -> skipped (left pending)")
                    break
                if choice in ("p", "play"):
                    cue(vi)
                    continue
                if choice in ("n", "next"):
                    vi = (vi + 1) % len(violations)
                    print(f"  -> violation {vi + 1}/{len(violations)}")
                    cue(vi)
                    continue
                if choice in ("q", "quit"):
                    print("  -> quit")
                    raise KeyboardInterrupt
                print("  ? unrecognised")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if mpv is not None:
            mpv.stop()

    print(f"\nReviewed: {approved} approved, {rejected} rejected, "
          f"{skipped} skipped, {len(pending) - approved - rejected - skipped} "
          f"not reached.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Human review UI for the violation queue")
    p.add_argument("--data", default="./data", help="DATA_DIR (default ./data)")
    p.add_argument("--no-mpv", action="store_true",
                   help="don't launch mpv; print the manual play command instead")
    args = p.parse_args(argv)
    return review_queue(Path(args.data), use_mpv=not args.no_mpv)


if __name__ == "__main__":
    sys.exit(main())
