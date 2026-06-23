# SOLUTION — Local Dashcam Violation Pipeline

A self-contained design/spec for a **local, file-based** system that ingests
dashcam footage and uses a local vision-language model (VLM) on an RTX 5070 to
**propose other drivers' road-rule violations** for human approval.

Drop this in the repo root. The accompanying scaffold (`docker-compose.yml`,
`worker/`) already implements the core loop; this document is the "why" and the
"what's left."

---

## 1. Goal & scope

- **Input:** dashcam SD card, dumped daily.
- **Output:** a file-based queue of candidate violations *by other road users*,
  each with frame timestamp, plate (when legible), and a one-line description,
  awaiting human approve/reject.
- **Constraints (hard):**
  - Runs **fully locally** under CachyOS on an RTX 5070 (12 GB). No cloud.
  - **No database** — all findings, records, and statuses are plain files
    (SQLite only if cross-clip *queries* are ever needed; see §6).
  - **Reproducible** via Docker / Docker Compose.
  - `mpv` is the playback tool for review.

### Two decisions that shaped everything

1. **Others' violations, not the camera car.** This makes the offending
   vehicle's **licence plate** part of the record (not optional), and makes
   **evidence integrity** matter — the original clip is kept byte-for-byte
   untouched; all analysis runs on copies/frames.
2. **SD-card dump ingestion.** Dashcams loop-record and **reuse filenames**, so
   deduplication must be by **file content hash**, never by name. Re-dumping the
   same card must be a no-op.

---

## 2. Hardware constraint & model choice

The desktop **RTX 5070 has 12 GB of GDDR7** (192-bit, ~672 GB/s, Blackwell).
That puts it in the 7B–13B vision-model tier.

**Chosen VLM: Qwen3-VL-8B** (served as `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`).
- Native video understanding with timestamp-grounded event localisation — well
  suited to "find the moment a violation happens."
- The 30B-A3B MoE variant wants ~14–16 GB → too tight once frames are added.

**Serving: vLLM, not Ollama** (this changed during bring-up — see below).
vLLM runs the *entire* model, including the vision encoder, on the GPU and serves
the same OpenAI-compatible `/v1/chat/completions`. Measured on this box: **~0.2 s
per 8-frame call warm** (≈4 s/clip end-to-end incl. frame extraction).

> **Why not Ollama (the original plan).** Ollama was tried first. On a 12 GB
> card it refuses to offload Qwen3-VL's multimodal projector to the GPU
> (`--no-mmproj-offload`, `reason=limited-vram`) and runs the vision encoder
> (`CLIP using CPU backend`) on the CPU — **~16 s per frame, ~2 min/clip**, GPU
> idle at ~0–2 %. Confirmed at context 2048/4096/8192/16384 and with q8 KV +
> flash attention; none moved it to GPU. It is *not* a Blackwell support gap
> (the CUDA build includes sm_120). Ollama remains available as an opt-in
> fallback (`docker compose --profile ollama`), but it is CPU-bound here.

> **Fitting 8B in 12 GB under vLLM.** bf16 (~16 GB) and FP8 (~10.6 GB) both OOM
> once KV cache + GPU vision activations are added. The **AWQ 4-bit** quant of
> the 8B (~7.6 GB) fits with room to spare; `Qwen/Qwen3-VL-4B-Instruct-FP8`
> (~6 GB) is the fallback if more headroom is wanted. Note this is a *desktop*
> GPU — KDE/Chrome/VSCode hold ~1.8 GB, so `--gpu-memory-utilization` is kept at
> 0.80 to coexist (raise toward 0.92 for a headless nightly run).

> Blackwell needs a recent stack: NVIDIA driver 570+ (this box: 610), CUDA 12.8+,
> and the NVIDIA Container Toolkit. The worker uses a CUDA 12.8 base; vLLM uses
> the `cu129-nightly` image (sm_120 kernels). Older CUDA images won't see the card.

---

## 3. Landscape — reuse, don't reinvent

No turnkey "dashcam folder → local VLM → file-based approval queue" tool exists.
The reusable pieces:

| Component | Use it for | Notes |
|---|---|---|
| **Predator** (connervieira/Predator) | Ingestion, **ALPR**, GPS/GPX correlation, object logging | Open source, offline, file-based logging. Does *not* reason about violations — lift its plate reader. |
| **Frigate NVR** | watch-folder + object-detect + review-UI pattern | Built for live RTSP cameras; against its grain for batch dashcam offload. |
| **YOLO + tracking + rules** projects (HuandongChang/Traffic_Violation_Detection; arXiv 2311.16179, six infractions) | Reference logic for stop-sign / red-light / following-distance rules | Proof-of-concept, not production. |
| **DashcamCleaner** | Auto-blur plates & faces | Only if you ever export/share a clip. |

The missing glue — orchestration, VLM reasoning, file-state machine, review — is
what this project builds.

---

## 4. Architecture

Six idempotent, file-driven stages. Any stage can be re-run safely; state is
read from and written to `manifest.json`.

```
SD card ─ingest─> organize ─detect (YOLO gate)─> review (VLM) ─> pending_review ─> human approve/reject
 (ro)    dedup by   folder/clip   keep clips with    structured        queue          status written
         content    + manifest    traffic objects    verdict + plate                  back to file
         hash
```

1. **Ingest** — scan the read-only SD mount, hash each video, skip hashes already
   in the dedup ledger, copy new ones in (set read-only), write initial manifest.
2. **Organize** — one folder per clip: `data/clips/<date>/<clip_id>/`. (In the
   scaffold this is folded into ingest.)
3. **Detect (gate)** — cheap YOLO pass over sampled frames; keep the clip only if
   trigger objects (vehicles, lights, signs, pedestrians) appear enough times.
   Most footage is empty road — this is what keeps VLM cost to minutes/day.
4. **Review (VLM)** — sample N frames from candidate clips, send with a
   rules-derived prompt, parse a structured JSON verdict, keep violations above a
   confidence floor → status `pending_review`.
5. **Human review** — the queue is every manifest with `status == pending_review`;
   approve/reject writes status back.
6. **Export/archive** — approved records: original clip + JSON + (optional) blurred
   share copy. Pruning per retention policy.

---

## 5. File-state design (the "no DB" part)

`manifest.json` per clip is the single source of truth. Example after review:

```json
{
  "clip_id": "a1b2c3d4e5f6",
  "sha1": "a1b2c3d4e5f6...",
  "source_name": "FILE0123.MP4",
  "captured_date": "2026-06-23",
  "ingested_at": "2026-06-23T20:04:11",
  "video": "original.mp4",
  "candidate": true,
  "yolo_hits": 7,
  "violations": [
    {
      "rule_id": "red_light",
      "confidence": 0.82,
      "frame_time": 14.5,
      "plate": "01KG123ABC",
      "description": "Silver sedan crosses the stop line ~1s after the light turns red."
    }
  ],
  "vlm": { "...": "raw model output, kept for audit" },
  "reviewed_at": "2026-06-23T20:09:03",
  "status": "pending_review"
}
```

**Status state machine:**
```
ingested ─> candidate ──> pending_review ──> approved
        └─> cleared       └─> cleared        └─> rejected
```

**Layout:**
```
data/
  ingested_hashes.txt        # plain-file dedup ledger (one sha1 per line)
  clips/
    2026-06-23/
      a1b2c3d4e5f6/
        original.avi         # read-only, never modified (real extension kept)
        manifest.json        # state + verdict
        thumbs/              # extracted candidate frames (optional)
```

**When SQLite earns its place:** only when you need cross-clip *queries* (e.g.
"every red-light event within this GPS box last month"). Until then, globbing
manifests is fast into the thousands and keeps the system grep-able and
git-friendly.

---

## 6. Two-tier detection rationale

Pure VLM on every frame of every clip is wasteful and slow. Pure YOLO+rules is
brittle on judgment calls (was the light actually red? did the car actually fail
to stop, or just roll slowly?).

So: **YOLO is a cheap gate**, the **VLM is the judgment**. YOLO discards empty
road; the VLM only ever sees clips that *might* contain a violation and decides
whether one actually occurred. Disable the gate (`use_yolo_prefilter: false`) to
send everything to the VLM during early testing.

---

## 7. VLM interface

- **Endpoint:** OpenAI-compatible `POST /v1/chat/completions` (default vLLM;
  Ollama opt-in — both speak the same API).
- **Message:** one text block (prompt) + N image blocks (base64 JPEG data URLs,
  sampled across the candidate window).
- **Prompt structure:** role framing ("review OTHER road users, not the camera
  car") + the rule list from `config.yaml` + "only report what you can see" +
  "transcribe legible plates" + a strict JSON-only output contract.
- **Output schema:**
  ```json
  {"violations":[{"rule_id":"","confidence":0.0,"frame_time":0.0,"plate":"","description":""}]}
  ```
- Parse defensively (strip ``` fences, tolerate parse failure → store raw for
  audit, treat as no-violation).

Rules live in `worker/config.yaml` and are the main accuracy lever — edit
wording, add/remove rules, tune `vlm.frames_per_window` and `vlm.min_confidence`.

---

## 8. Stack & reproducibility

- **`docker-compose.yml`** — services:
  - `vllm` (GPU-reserved) serving Qwen3-VL on the GPU; healthchecked so the
    worker waits until it is ready. Reuses a `dashcam_hf_cache` volume for weights.
  - `worker` (GPU-reserved for YOLO) running the pipeline; SD mounted read-only,
    `data/` mounted read-write; runs as the host UID/GID so `data/` stays yours.
  - `ollama` (opt-in, `--profile ollama`) — CPU-bound vision here; see §2.
- **CachyOS setup:**
  ```bash
  sudo pacman -S nvidia-open nvidia-container-toolkit docker docker-buildx docker-compose
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  sudo usermod -aG docker $USER   # then re-login (or `newgrp docker`)
  ```
- **First run:**
  ```bash
  cp .env.example .env            # SD_MOUNT, DATA_DIR, HOST_UID/GID, VLLM_GPU_UTIL
  docker compose up -d vllm       # first boot downloads ~7.6 GB weights + JITs kernels (~2 min)
  docker compose run --rm worker run
  docker compose run --rm worker list
  ```
- **Daily schedule** (host cron / systemd timer):
  ```
  0 20 * * *  cd /path/to/repo && docker compose up -d vllm && docker compose run --rm worker run
  ```

---

## 9. Review with mpv

Jump to the flagged moment:
```bash
mpv --start=$(jq -r '.violations[0].frame_time' manifest.json) "$(jq -r '.video' manifest.json)"
```
(The ingested original keeps its real extension, e.g. `original.avi`.)
For the review UI, run mpv with `--input-ipc-server=/tmp/mpvsock` and drive
seek/play over its JSON IPC socket, writing `approved`/`rejected` back to the
manifest.

---

## 10. Honest limits

- VLM output is a **proposal, not evidence** — hence the human-approval step.
  Keep the raw model output in the manifest for audit.
- Expect false positives from glare, occlusion, ambiguous signals, night. Tune
  the YOLO gate and `vlm.min_confidence` to control queue volume.
- **Evidence integrity:** never modify the original; do all work on copies/frames.
- **Legal:** whether approved records can be acted on — reported, and to whom —
  varies by jurisdiction. Some countries run official citizen-report portals;
  some prohibit citizen enforcement or publishing dashcam footage with plates.
  Confirm the local rule before building the export/reporting stage.
  *(Open question — see §12.)*

---

## 11. Build status

**Done & validated on real footage (TR10 card, RTX 5070):** ingest +
content-hash dedup (re-dump is a no-op), organize with original-extension
preservation + read-only originals, filename-derived capture dates, YOLO gate on
GPU, **VLM verdict via vLLM on GPU** (~4 s/clip), file-based `pending_review`
queue, host-user-owned `data/`, daily-runnable via Compose (vllm healthcheck →
worker).

**Next tasks (suggested order):**
1. **Review UI + mpv IPC** — close the human-approval loop (list pending →
   play at `frame_time` → write status). The one manual gap today.
2. **VLM prompt + rules tuning** — biggest accuracy lever; iterate on wording and
   frames-per-window against real clips. (Early subset all `cleared`; needs a
   labelled positive to calibrate `min_confidence` and prompt wording.)
3. **Dedicated ALPR** — replace VLM plate-guessing with a real plate reader
   (lift Predator's) for reliable identification.
4. **GPS/GPX correlation** — stamp each record with location from embedded telemetry.
5. **Export stage** — approved record bundle (+ optional plate/face blur via
   DashcamCleaner if sharing).

---

## 12. Open questions

- **Jurisdiction** — which country? Determines whether/where approved records can
  be reported, which shapes the export stage. *(Still open.)*
- ~~**Dashcam telemetry format**~~ — **Answered:** the unit is a **TR10**; each
  `.avi` carries GPS/accelerometer telemetry in **subtitle tracks**
  (`gpsStat=1, accelerStat=1`). Usable for §11 task 4 / speed-based rules — needs
  a subtitle-track parser.
- ~~**Cameras**~~ — **Answered:** **front + rear**, both muxed as two 1080p30
  video streams in one `.avi`. The pipeline currently samples the default (front)
  stream; rear is available as a second stream for rear-facing rules.

---

## Repo layout

```
.
├── README.md                # run instructions
├── SOLUTION.md              # this file
├── docker-compose.yml
├── .env.example
└── worker/
    ├── Dockerfile
    ├── requirements.txt
    ├── config.yaml          # model, YOLO triggers, rules, thresholds
    └── pipeline.py          # ingest / detect / review / queue
```
