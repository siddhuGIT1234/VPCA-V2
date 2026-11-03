# VPCA Starter — Progressive Delivery Prototype

Python server streams a scientific point dataset level by level to a VTK.js web client.

```
server/   Python: dataset, levels (delivery methods), FastAPI WebSocket server
client/   JavaScript: VTK.js viewer that appends points as levels arrive
logs/     One .jsonl file per session (timings, FPS, frame stalls)
```

## First-time setup (Windows, PowerShell)

```powershell
# 1. Python side
cd vpca-starter\server
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python make_dataset.py              # creates server\data\synthetic.npz (200k points)

# 2. JavaScript side (new terminal)
cd vpca-starter\client
npm install
```

## Run it (two terminals)

```powershell
# Terminal 1 — server
cd vpca-starter\server
.venv\Scripts\Activate.ps1
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — client
cd vpca-starter\client
npm run dev          # or: npm run dev:https  (for Quest)
```

Open **http://localhost:5173**, set a delay, click **Start stream**.
Each session writes a log to `logs/`.

## Using your own VTK data

Click **Choose file…** in the panel and pick a `.vtp`, `.vtu`, `.vti`, `.vtk` (legacy) or
`.vtm` file. The server converts it to point records and it appears in the **Dataset** list
(and stays there after restarts — converted files live in `server/data/`).

| File type | What becomes a point | Attributes |
|---|---|---|
| `.vtp` PolyData (surfaces, point clouds) | each vertex | point arrays |
| `.vtu` UnstructuredGrid (FEM/CFD meshes) | each node | point arrays + cell arrays averaged onto nodes |
| `.vti` ImageData (regular volumes, CT/simulation grids) | each grid point | point arrays |
| `.vtk` legacy / `.vtm` multiblock | as above | as above |

Vectors (e.g. `velocity`) become `_x`, `_y`, `_z` and `_magnitude` columns. Files with
more than 2,000,000 points are randomly subsampled (change `MAX_POINTS` in `app.py`);
the panel shows when that happened. Everything is rendered as colored points.

## How it works

1. `make_dataset.py` — 3 spatial blobs + a 1% **rare group** (label 3) with extreme
   temperature/pressure, hidden inside the blobs. Ground truth for rare-group recall.
2. `levels.py` — a delivery method returns an **ordering** of records; levels are
   prefixes of it, so `L0 ⊂ L1 ⊂ ...` holds automatically and all methods get the
   same point budgets. Only `random` exists today.
3. `app.py` — `/manifest` (metadata) and `/stream` (WebSocket). Each chunk is a JSON
   header followed by binary float32 rows: `[record_id, x, y, z, attr_0..attr_3]`.
   Levels are split into 512 KB chunks to avoid frame stalls.
4. `client/src/main.js` — preallocates buffers for the whole dataset, appends each
   chunk, re-points VTK.js at the larger view (no rebuild), logs parse/update time
   and FPS back to the server.

## Next steps

- [ ] Add `order_spatial` and `order_aahr` in `levels.py` (MiniBatchKMeans) → they appear in `METHODS`
- [ ] Metric script: rare-group recall, attribute Wasserstein distance per level, from `data/` + method
- [x] WebXR "Enter VR" button
- [ ] Controller interaction in VR (grab/rotate/scale)
- [ ] Run on Quest 2 (needs HTTPS — see below) and profile max points at 72 Hz
- [ ] ViVo-style / Theia-style view-dependent baselines, greedy oracle, then VPCA

## Running on Meta Quest 2

WebXR (VR mode) only works over HTTPS. The client dev server can serve HTTPS with a
self-signed certificate, and it forwards `/api` and `/ws` to the Python server, so the
Quest only needs to reach port 5173.

1. Laptop and Quest on the **same Wi-Fi**.
2. Terminal 1: start the Python server as usual (`uvicorn app:app --reload --port 8000`).
3. Terminal 2: `npm run dev:https` and note the **Network** address, e.g. `https://192.168.0.90:5173`.
   If Windows Firewall asks about Node.js, allow it on **Private networks**.
4. On the Quest, open **Meta Quest Browser**, type that address, then
   **Advanced → Proceed** past the certificate warning (it's your own laptop).
5. In the 2D page: pick a dataset, raise **Point size** to ~4–6, click **Start stream**,
   then click **Enter VR**. Walk around the data; press the Quest menu button to exit.

Frame timings recorded inside VR are logged with `"xr": true` in `logs/*.jsonl` —
these are the Quest measurements for the paper.

Alternative over USB (needs Developer Mode + adb): `npm run dev`, then
`adb reverse tcp:5173 tcp:5173` and open `http://localhost:5173` on the Quest
(localhost counts as secure, no certificate warning).
