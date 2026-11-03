"""Progressive delivery server.

GET  /datasets                         -> list of available datasets
POST /upload  (multipart file)         -> convert .vtp/.vtu/.vti/.vtk/.vtm, register it
GET  /manifest?dataset=..&method=..    -> dataset metadata (JSON)
WS   /stream?dataset=..&method=..&delay_ms=0
     server -> client: {"type":"level", ...} (JSON) then that chunk's bytes (binary)
                       {"type":"done"} at the end
     client -> server: {"type":"log", ...} lines, saved to logs/<session>.jsonl

Run:  uvicorn app:app --reload --port 8000
"""
import asyncio
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from levels import build_levels
from regions import build_regions, point_region_ids, describe_region

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_DIR = ROOT.parent / "logs"
for d in (DATA_DIR, UPLOAD_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHUNK_RECORDS = 16_384        # 16k records x stride floats x 4 bytes per message
MAX_POINTS = 15_000_000       # retains all 13,983,769 Waldeck points
ALLOWED = {".vtp", ".vtu", ".vti", ".vtk", ".vtm"}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_datasets = {}   # dataset_id -> {"data": {...}, "names": [...], "info": {...}}
_levels = {}     # (dataset_id, method) -> (levels, sizes)


def _register(dataset_id, npz_path):
    raw = np.load(npz_path, allow_pickle=False)
    info = json.loads(str(raw["info"])) if "info" in raw.files else {}
    _datasets[dataset_id] = {
        "data": {k: raw[k] for k in ("record_id", "pos", "att", "label")},
        "names": [str(a) for a in raw["attribute_names"]],
        "info": info,
    }
    for key in [k for k in _levels if k[0] == dataset_id]:
        del _levels[key]


def _load_all():
    for npz in sorted(DATA_DIR.glob("*.npz")):
        try:
            _register(npz.stem, npz)
        except Exception as e:   # a broken file shouldn't stop the server
            print(f"skipping {npz.name}: {e}")
    if not _datasets:
        print("No datasets yet. Run  python make_dataset.py  or upload a VTK file.")


_load_all()


def get_dataset(dataset_id):
    if dataset_id not in _datasets:
        raise HTTPException(404, f"Unknown dataset '{dataset_id}'")
    return _datasets[dataset_id]


def get_levels(dataset_id, method):
    key = (dataset_id, method)
    if key not in _levels:
        _levels[key] = build_levels(get_dataset(dataset_id)["data"], method)
    return _levels[key]


def get_regions(dataset_id):
    ds = get_dataset(dataset_id)
    if "regions" not in ds:
        ds["regions"] = build_regions(ds["data"]["pos"])
    return ds["regions"]


@app.get("/point-region")
def point_region(dataset: str, record_id: int = Query(..., ge=0)):
    ds = get_dataset(dataset)
    pos = ds["data"]["pos"]
    if record_id >= len(pos):
        raise HTTPException(404, "Point record does not exist in this dataset")
    info = get_regions(dataset)
    rid = int(point_region_ids(pos[record_id:record_id + 1],
                              info["minimum"], info["maximum"], info["grid"])[0])
    original_id, id_kind = record_id, "dataset record index"
    # Compatible with the already-imported Waldeck NPZ; no re-upload required.
    # OriginalPointId values in this 13.98M-point file are exact in float32.
    if "OriginalPointId" in ds["names"]:
        value = float(ds["data"]["att"][record_id, ds["names"].index("OriginalPointId")])
        if np.isfinite(value) and value >= 0 and value.is_integer():
            original_id = int(value)
            id_kind = "stored OriginalPointId attribute"
    return {"dataset_id": dataset, "record_id": record_id,
            "original_point_id": original_id, "id_kind": id_kind,
            "position": pos[record_id].astype(float).tolist(),
            **describe_region(rid, info)}


@app.get("/datasets")
def datasets():
    return [
        {"id": k, "points": int(len(v["data"]["pos"])), "attributes": v["names"], **v["info"]}
        for k, v in sorted(_datasets.items())
    ]


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    from loader import load_vtk   # imported here so the server starts even without pyvista

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use one of {sorted(ALLOWED)}")

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(file.filename).stem)[:40] or "dataset"
    dataset_id = f"{stem}{suffix.replace('.', '_')}"
    src = UPLOAD_DIR / f"{dataset_id}{suffix}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t = time.time()
    try:
        data, info = load_vtk(str(src), max_points=MAX_POINTS)
    except Exception as e:
        raise HTTPException(400, f"Could not read {file.filename}: {e}")
    info.update(source_file=file.filename, convert_seconds=round(time.time() - t, 2))

    npz_path = DATA_DIR / f"{dataset_id}.npz"
    np.savez(npz_path, **data, info=np.array(json.dumps(info)))
    _register(dataset_id, npz_path)
    return {"id": dataset_id, "points": int(len(data["pos"])),
            "attributes": [str(a) for a in data["attribute_names"]], **info}


@app.get("/manifest")
def manifest(dataset: str = "synthetic", method: str = "random"):
    ds = get_dataset(dataset)
    levels, sizes = get_levels(dataset, method)
    att = ds["data"]["att"]
    # 1st-99th percentile color range so a few extreme values don't wash out the colors
    lo = np.percentile(att, 1, axis=0)
    hi = np.percentile(att, 99, axis=0)
    # Near-constant fields: widen the range so float rounding noise isn't
    # stretched across the whole colormap (shows as fake bands otherwise).
    scale = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1e-6)
    flat = (hi - lo) < 1e-4 * scale
    lo = np.where(flat, lo - 0.01 * scale, lo)
    hi = np.where(flat, hi + 0.01 * scale, hi)
    return {
        "dataset_id": dataset,
        "method": method,
        "record_count": int(len(ds["data"]["pos"])),
        "attribute_names": ds["names"],
        "attribute_ranges": [[float(a), float(b)] for a, b in zip(lo, hi)],
        "stride": 4 + len(ds["names"]),          # floats per record on the wire
        "number_of_levels": len(levels),
        "cumulative_sizes": sizes,
        "bytes_per_level": [int(l.nbytes) for l in levels],
        "info": ds["info"],
        "regions": get_regions(dataset),
    }


@app.websocket("/stream")
async def stream(ws: WebSocket, dataset: str = "synthetic", method: str = "random",
                 delay_ms: int = Query(0, ge=0)):
    await ws.accept()
    if dataset not in _datasets:
        await ws.send_text(json.dumps({"type": "error", "message": f"Unknown dataset '{dataset}'"}))
        await ws.close()
        return
    session = time.strftime("%Y%m%d-%H%M%S") + f"_{dataset}_{method}_d{delay_ms}"
    log_path = LOG_DIR / f"{session}.jsonl"
    levels, sizes = get_levels(dataset, method)

    async def receive_logs():
        with open(log_path, "a") as f:
            while True:
                msg = json.loads(await ws.receive_text())
                msg["server_t"] = time.time()
                f.write(json.dumps(msg) + "\n")
                f.flush()

    reader = asyncio.create_task(receive_logs())
    try:
        for i, rows in enumerate(levels):
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)   # crude network emulation
            # Large levels go out in chunks so the client never parses
            # megabytes in one frame (avoids stalls on Quest).
            chunks = range(0, len(rows), CHUNK_RECORDS)
            for c, start in enumerate(chunks):
                part = rows[start:start + CHUNK_RECORDS]
                await ws.send_text(json.dumps({
                    "type": "level", "level": i, "chunk": c, "chunks": len(chunks),
                    "count": int(len(part)), "bytes": int(part.nbytes),
                    "server_send_t": time.time(),
                }))
                await ws.send_bytes(part.tobytes())
                await asyncio.sleep(0)
        await ws.send_text(json.dumps({"type": "done"}))
        await reader          # keep the socket open to collect client logs
    except WebSocketDisconnect:
        pass
    finally:
        reader.cancel()
        print(f"session {session} closed, log: {log_path}")
