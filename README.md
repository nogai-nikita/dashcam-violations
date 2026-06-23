# Dashcam Violation Pipeline

A **local, file-based** pipeline that ingests dashcam footage and uses a local
vision-language model (Qwen3-VL-8B, served on the GPU by **vLLM**) on an RTX 5070
to **propose other drivers' road-rule violations** for human approval. No cloud,
no database — every finding is a plain `manifest.json` file.

See [SOLUTION.md](SOLUTION.md) for the full design rationale — including §2 on why
vLLM replaced Ollama here (Ollama runs Qwen3-VL's vision encoder on CPU on a 12 GB
card: ~2 min/clip vs ~4 s/clip with vLLM).

```
SD card ─ingest─> organize ─detect (YOLO gate)─> review (VLM) ─> pending_review ─> human approve/reject
 (ro)    dedup by   folder/clip   keep clips with    structured        queue
         content    + manifest    traffic objects    verdict + plate
```

## Layout

```
.
├── docker-compose.yml      # vllm + worker (GPU-reserved); ollama opt-in profile
├── .env.example            # SD_MOUNT, DATA_DIR, HOST_UID/GID, VLLM_GPU_UTIL
├── review_ui.py            # host-side mpv review UI (approve/reject the queue)
└── worker/
    ├── Dockerfile          # CUDA 12.8 base (Blackwell), torch cu128, ultralytics
    ├── requirements.txt
    ├── config.yaml         # model, YOLO triggers, rules, thresholds  <-- accuracy lever
    └── pipeline.py         # ingest / detect / review / list / run
```

Output lives under `DATA_DIR`:

```
data/
  ingested_hashes.txt        # dedup ledger (one sha1 per line)
  clips/2026-06-20/<clip_id>/
    original.avi             # read-only, never modified (real extension preserved)
    manifest.json           # state + verdict
```

## Run with Docker (intended path)

Prereqs on CachyOS:

```bash
sudo pacman -S nvidia-open nvidia-container-toolkit docker docker-buildx docker-compose
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER    # then re-login (or `newgrp docker`)
```

First run:

```bash
cp .env.example .env                       # edit SD_MOUNT, DATA_DIR, HOST_UID/GID
docker compose up -d vllm                  # first boot: ~7.6 GB weight download + ~2 min kernel JIT
docker compose run --rm worker run         # ingest + detect + review
docker compose run --rm worker list        # show the pending-review queue
```

`vllm` is healthchecked, so `worker` waits until the model is ready. The 8B is
served as the AWQ 4-bit quant to fit 12 GB; tune `VLLM_GPU_UTIL` in `.env`
(0.80 to coexist with a desktop, up to ~0.92 headless).

Daily (host cron / systemd timer):

```
0 20 * * *  cd /path/to/repo && docker compose up -d vllm && docker compose run --rm worker run
```

## Run natively (no Docker)

Useful for fast iteration on this machine. Needs `ffmpeg`/`ffprobe` and a
reachable VLM endpoint (e.g. the `vllm` container on `:8000`, or any
OpenAI-compatible server). `ultralytics`/`torch` are only needed when the YOLO
gate is enabled.

```bash
python3 -m venv .venv && source .venv/bin/activate    # bash; fish: source .venv/bin/activate.fish
pip install -r worker/requirements.txt                # or: pip install requests pyyaml  (VLM-only)

# point the worker at the running vLLM server and pass paths explicitly
export VLM_BASE_URL=http://localhost:8000 VLM_MODEL=qwen3-vl
python worker/pipeline.py run  --sd /path/to/sd --data ./data
python worker/pipeline.py list --data ./data
```

`VLM_BASE_URL`/`VLM_MODEL` override `config.yaml` (and the compose stack sets
them). `OLLAMA_BASE_URL` is still honoured for the opt-in Ollama path.

To skip YOLO during early testing, set `detect.use_yolo_prefilter: false` in
`worker/config.yaml` — then every ingested clip goes straight to the VLM and
neither torch nor ultralytics is imported.

## Stages (all idempotent — safe to re-run)

| Command | Does |
|---|---|
| `pipeline.py ingest` | hash + dedup + copy + organize, write initial manifest |
| `pipeline.py detect` | YOLO gate → `candidate` or `cleared` |
| `pipeline.py review` | VLM judgment → `pending_review` or `cleared` |
| `pipeline.py list`   | print the pending-review queue |
| `pipeline.py run`    | ingest + detect + review |

`--force` re-runs detect/review over already-processed clips (e.g. after tuning
`config.yaml`). Re-dumping the same SD card is a no-op (content-hash dedup).

## Review the queue (human approve/reject)

`review_ui.py` walks every `pending_review` clip, plays it in mpv jumped to the
flagged moment, and writes `approved`/`rejected` back to the manifest. It is a
**host tool** (drives the host's mpv + display), stdlib-only — run it with the
system `python3`, no venv:

```bash
python3 review_ui.py --data ./data        # opens mpv; a/r/s/p/n/q per clip
python3 review_ui.py --data ./data --no-mpv   # decide from the stills/CLI only
```

Keys: `[a]`pprove · `[r]`eject · `[s]`kip · `[p]`lay again · `[n]`ext violation ·
`[q]`uit. A decision appends a `human_review` block (decision, reviewer,
timestamp, note) for audit; the original clip is never touched.

Or just jump to the moment by hand:

```bash
M=data/clips/2026-06-20/<clip_id>/manifest.json
mpv --start=$(jq -r '.violations[0].frame_time' "$M") \
    "$(dirname "$M")/$(jq -r '.video' "$M")"
```

## Status state machine

```
ingested ─> candidate ──> pending_review ──> approved
        └─> cleared       └─> cleared        └─> rejected
```

## Honest limits

VLM output is a **proposal, not evidence** — hence the human-approval step; the
raw model output is kept in each manifest for audit. Expect false positives from
glare, occlusion, night. Tune the YOLO gate and `vlm.min_confidence` to control
queue volume. Never modify the original clip. Whether approved records can be
**reported** depends on your jurisdiction — confirm before building export.
