# Dashcam Violation Pipeline

A **local, file-based** pipeline that ingests dashcam footage and uses a local
vision-language model (Ollama + `qwen3-vl:8b`) on an RTX 5070 to **propose other
drivers' road-rule violations** for human approval. No cloud, no database — every
finding is a plain `manifest.json` file.

See [SOLUTION.md](SOLUTION.md) for the full design rationale.

```
SD card ─ingest─> organize ─detect (YOLO gate)─> review (VLM) ─> pending_review ─> human approve/reject
 (ro)    dedup by   folder/clip   keep clips with    structured        queue
         content    + manifest    traffic objects    verdict + plate
```

## Layout

```
.
├── docker-compose.yml      # ollama + worker, both GPU-reserved
├── .env.example            # SD_MOUNT, DATA_DIR
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
  clips/2026-06-23/<clip_id>/
    original.mp4             # read-only, never modified
    manifest.json           # state + verdict
```

## Run with Docker (intended path)

Prereqs on CachyOS:

```bash
sudo pacman -S nvidia-open nvidia-container-toolkit docker docker-compose
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo systemctl enable --now docker
```

First run:

```bash
cp .env.example .env                       # edit SD_MOUNT, DATA_DIR
docker compose up -d ollama
docker exec dv-ollama ollama pull qwen3-vl:8b
docker compose run --rm worker             # ingest + detect + review
docker compose run --rm worker list        # show the pending-review queue
```

Daily (host cron / systemd timer):

```
0 20 * * *  cd /path/to/repo && docker compose run --rm worker
```

## Run natively (no Docker)

Useful for fast iteration on this machine. Needs `ffmpeg`/`ffprobe` and a local
Ollama; `ultralytics`/`torch` are only needed when the YOLO gate is enabled.

```bash
python3 -m venv .venv && source .venv/bin/activate    # bash; fish: source .venv/bin/activate.fish
pip install -r worker/requirements.txt                # or: pip install requests pyyaml  (VLM-only)

# point the VLM at a local ollama and pass paths explicitly
export OLLAMA_BASE_URL=http://localhost:11434
python worker/pipeline.py run  --sd /path/to/sd --data ./data
python worker/pipeline.py list --data ./data
```

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

## Review a flagged clip with mpv

```bash
M=data/clips/2026-06-23/<clip_id>/manifest.json
mpv --start=$(jq -r '.violations[0].frame_time' "$M") "$(dirname "$M")/original.mp4"
```

The interactive mpv-IPC approve/reject loop (writing `approved`/`rejected` back
to the manifest) is the next task — see [SOLUTION.md](SOLUTION.md) §11.

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
