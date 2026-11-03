// Thin client: receive levels, append points, color by attribute, log timings.
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkWebXRRenderWindowHelper from '@kitware/vtk.js/Rendering/WebXR/RenderWindowHelper';
import { XrSessionTypes } from '@kitware/vtk.js/Rendering/WebXR/RenderWindowHelper/Constants';
import { regionForCoords } from './regions';
import { setupPointSelection } from './selection';

// Both go through the Vite dev server's proxy (see vite.config.js).
const SERVER = '/api';
const WS_SERVER = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
const $ = (id) => document.getElementById(id);

// ---------- VTK.js scene ----------
const fsrw = vtkFullScreenRenderWindow.newInstance({
  rootContainer: $('view'),
  background: [0.07, 0.07, 0.07],
});
const renderer = fsrw.getRenderer();
const renderWindow = fsrw.getRenderWindow();
const polyData = vtkPolyData.newInstance();
const mapper = vtkMapper.newInstance({ scalarVisibility: true });
mapper.setInputData(polyData);
// vtk.js's default map is red=low, blue=high. Flip it so blue=low, red=high
// (hot / high-pressure / extreme values show up red, as people expect).
mapper.getLookupTable().setHueRange(0.667, 0.0);
const actor = vtkActor.newInstance();
actor.setMapper(mapper);
actor.getProperty().setPointSize(2);
renderer.addActor(actor);
window.vpca = { renderer, renderWindow, fsrw }; // handy for debugging in the browser console

// ---------- State ----------
let manifest = null;
let positions, verts, attrs; // preallocated for the full dataset
let received = 0;            // records delivered so far
let ws = null;
let t0 = 0;
let pendingHeader = null;
let inXR = false;           // true while the headset session is active
let recordIds, regionIds;
let complete = false, epoch = 0;

function log(event, extra = {}) {
  const msg = { type: 'log', event, t: performance.now() - t0, received, ...extra };
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function allocate(m) {
  const n = m.record_count;
  positions = new Float32Array(n * 3);
  verts = new Uint32Array(n * 2);      // vtk cell layout: [1, id, 1, id, ...]
  attrs = m.attribute_names.map(() => new Float32Array(n));
  received = 0;
  recordIds = new Uint32Array(n);
  regionIds = new Uint16Array(n);
}

function applyColor() {
  if (!manifest) return;
  const k = Number($('colorBy').value);
  const scalars = vtkDataArray.newInstance({
    name: manifest.attribute_names[k],
    values: attrs[k].subarray(0, received),
    numberOfComponents: 1,
  });
  polyData.getPointData().setScalars(scalars);
  mapper.setScalarRange(...manifest.attribute_ranges[k]);
  polyData.modified();
  renderWindow.render();
}

// Append one level's records (binary float32 rows) to the existing buffers.
function appendLevel(buffer) {
  const stride = manifest.stride;
  if (buffer.byteLength % (stride * 4) !== 0) {
    console.error('Bad payload size', buffer.byteLength, 'stride', stride);
    log('bad_payload', { bytes: buffer.byteLength });
    return;
  }
  const t = performance.now();
  const rows = new Float32Array(buffer);
  const count = rows.length / stride;
  for (let i = 0; i < count; i++) {
    const r = i * stride;
    const j = received + i;
    recordIds[j] = rows[r];
    positions[j * 3] = rows[r + 1];
    positions[j * 3 + 1] = rows[r + 2];
    positions[j * 3 + 2] = rows[r + 3];
    regionIds[j] = regionForCoords(rows[r + 1], rows[r + 2], rows[r + 3], manifest.regions);
    verts[j * 2] = 1;
    verts[j * 2 + 1] = j;
    for (let a = 0; a < attrs.length; a++) attrs[a][j] = rows[r + 4 + a];
  }
  received += count;
  const parseMs = performance.now() - t;

  // Point the VTK objects at the larger view of the same buffers (no copy).
  polyData.getPoints().setData(positions.subarray(0, received * 3), 3);
  polyData.getVerts().setData(verts.subarray(0, received * 2));
  applyColor();
  const h = pendingHeader;
  if (h.level === 0 && h.chunk === 0) {
    renderer.resetCamera();
    if (inXR) xrHelper.resetXRScene();
    log('first_view');
  }
  renderWindow.render();
  const updateMs = performance.now() - t;
  log('chunk_displayed', { level: h.level, chunk: h.chunk, count, bytes: buffer.byteLength, parse_ms: parseMs, update_ms: updateMs });
  if (h.chunk === h.chunks - 1) log('level_displayed', { level: h.level });
  updateStatus(`Level ${h.level + 1}/${manifest.number_of_levels}`);
}

function updateStatus(prefix) {
  $('status').innerHTML =
    `${prefix}<br>Points: ${received.toLocaleString()} / ${manifest.record_count.toLocaleString()}` +
    `<br>FPS: ${fps.toFixed(0)} &nbsp; worst frame: ${worstFrame.toFixed(1)} ms`;
}

// ---------- Frame-time meter ----------
let fps = 0, worstFrame = 0, frames = 0, lastSecond = performance.now(), lastFrame = performance.now();
function tick(now, fromXR = false) {
  if (inXR !== fromXR) {                  // only one loop counts at a time
    if (!fromXR) requestAnimationFrame(tick);
    return;
  }
  const dt = now - lastFrame;
  lastFrame = now;
  worstFrame = Math.max(worstFrame, dt);
  frames++;
  if (now - lastSecond >= 1000) {
    fps = (frames * 1000) / (now - lastSecond);
    if (ws && ws.readyState === WebSocket.OPEN) log('frames', { fps, worst_frame_ms: worstFrame, xr: inXR });
    if (manifest) updateStatus($('status').innerHTML.split('<br>')[0]);
    frames = 0; worstFrame = 0; lastSecond = now;
  }
  if (!fromXR) requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---------- Start a session ----------
async function start() {
  const thisEpoch = ++epoch;
  complete = false;
  selection.reset();
  if (ws) ws.close();
  const method = $('method').value;
  const delay = Number($('delay').value) || 0;
  t0 = performance.now();

  const dataset = $('dataset').value;
  if (!dataset) { $('status').textContent = 'No dataset. Upload a file or run make_dataset.py'; return; }
  $('status').textContent = 'Preparing stream and spatial regions...';
  const response = await fetch(`${SERVER}/manifest?dataset=${encodeURIComponent(dataset)}&method=${method}`);
  if (thisEpoch !== epoch) return;
  if (!response.ok) { $('status').textContent = `Cannot load dataset (${response.status})`; return; }
  manifest = await response.json();
  if (!manifest.regions) { $('status').textContent = 'Restart the updated Python server to enable region matching.'; return; }
  allocate(manifest);
  $('colorBy').innerHTML = manifest.attribute_names
    .map((name, i) => `<option value="${i}">${name}</option>`).join('');

  ws = new WebSocket(`${WS_SERVER}/stream?dataset=${encodeURIComponent(dataset)}&method=${method}&delay_ms=${delay}`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => log('session_start', { dataset, method, delay_ms: delay, ua: navigator.userAgent });
  ws.onmessage = (ev) => {
    if (thisEpoch !== epoch) return;
    if (typeof ev.data === 'string') {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'level') pendingHeader = msg;
      if (msg.type === 'done') {
        complete = received === manifest.record_count;
        log('all_levels_done'); updateStatus(complete ? 'Complete' : 'Incomplete stream');
        if (complete) selection.ready();
      }
      if (msg.type === 'error') $('status').textContent = msg.message;
    } else {
      appendLevel(ev.data);
    }
  };
  ws.onerror = () => { $('status').textContent = 'Cannot reach server on port 8000. Is it running?'; };
}

// ---------- Dataset list ----------
let datasetList = [];
async function refreshDatasets(selectId) {
  try {
    datasetList = await (await fetch(`${SERVER}/datasets`)).json();
  } catch {
    $('status').textContent = 'Cannot reach server on port 8000. Is it running?';
    return;
  }
  $('dataset').innerHTML = datasetList
    .map((d) => `<option value="${d.id}">${d.id} (${d.points.toLocaleString()} pts)</option>`).join('');
  if (selectId) $('dataset').value = selectId;
  showDatasetInfo();
}

function showDatasetInfo() {
  const d = datasetList.find((x) => x.id === $('dataset').value);
  if (!d) { $('dsInfo').textContent = ''; return; }
  let txt = `${d.attributes.length} attributes: ${d.attributes.join(', ')}`;
  if (d.source_type) txt = `${d.source_type} · ` + txt;
  if (d.subsampled) txt += ` · subsampled from ${d.points_in_file.toLocaleString()} points`;
  $('dsInfo').textContent = txt;
}

// ---------- Upload (XMLHttpRequest gives upload progress) ----------
function uploadFile(file) {
  const msg = $('uploadMsg'), bar = $('uploadBar');
  msg.classList.remove('err');
  bar.style.display = 'block';
  bar.value = 0;
  msg.textContent = `Uploading ${file.name} (${(file.size / 1e6).toFixed(1)} MB)…`;

  const form = new FormData();
  form.append('file', file);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `${SERVER}/upload`);
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    bar.value = (100 * e.loaded) / e.total;
    if (e.loaded === e.total) msg.textContent = 'Converting on server…';
  };
  xhr.onload = async () => {
    bar.style.display = 'none';
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch {}
    if (xhr.status !== 200) {
      msg.classList.add('err');
      msg.textContent = body.detail || `Upload failed (${xhr.status})`;
      return;
    }
    msg.textContent = `Loaded ${body.points.toLocaleString()} points, ` +
      `${body.attributes.length} attributes in ${body.convert_seconds}s` +
      (body.subsampled ? ` (subsampled from ${body.points_in_file.toLocaleString()})` : '');
    await refreshDatasets(body.id);
  };
  xhr.onerror = () => {
    bar.style.display = 'none';
    msg.classList.add('err');
    msg.textContent = 'Upload failed: cannot reach server';
  };
  xhr.send(form);
}

$('pick').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
  e.target.value = '';
});
$('dataset').addEventListener('change', showDatasetInfo);
$('pointSize').addEventListener('input', (e) => {
  $('psVal').textContent = e.target.value;
  actor.getProperty().setPointSize(Number(e.target.value));
  renderWindow.render();
});
// ---------- WebXR (Quest 2) ----------
const xrHelper = vtkWebXRRenderWindowHelper.newInstance({
  renderWindow: fsrw.getApiSpecificRenderWindow(),
  drawControllersRay: true,
});

async function setupVRButton() {
  const btn = $('vr');
  if (!navigator.xr) {
    btn.disabled = true;
    btn.textContent = window.isSecureContext ? 'VR not supported in this browser' : 'VR needs HTTPS (npm run dev:https)';
    return;
  }
  const ok = await navigator.xr.isSessionSupported('immersive-vr').catch(() => false);
  if (!ok) { btn.disabled = true; btn.textContent = 'No VR headset detected'; return; }
  btn.disabled = false;
  btn.textContent = 'Enter VR';
  btn.onclick = () => {
    if (!inXR) {
      xrHelper.startXR(XrSessionTypes.HmdVR);
    } else {
      xrHelper.stopXR();
    }
  };
}
xrHelper.onModified(() => {
  const session = xrHelper.getXrSession();
  const active = !!session;
  if (active === inXR) return;
  inXR = active;
  $('vr').textContent = inXR ? 'Exit VR' : 'Enter VR';
  log(inXR ? 'xr_enter' : 'xr_exit');
  if (inXR) {
    // Leaving with the Quest's own menu/home button ends the session directly.
    session.addEventListener('end', () => { if (inXR) xrHelper.stopXR(); });
    // The page's requestAnimationFrame pauses inside VR, so measure frames
    // with the XR session's own frame loop (these are the Quest numbers).
    const xrTick = (now) => {
      if (!inXR) return;
      tick(now, true);
      session.requestAnimationFrame(xrTick);
    };
    session.requestAnimationFrame(xrTick);
  }
});
setupVRButton();

$('resetView').addEventListener('click', () => {
  const cam = renderer.getActiveCamera();
  cam.setFocalPoint(0, 0, 0);       // look straight down -Z with Y up, then refit
  cam.setPosition(0, 0, 1);
  cam.setViewUp(0, 1, 0);
  renderer.resetCamera();
  renderWindow.render();
});
$('start').addEventListener('click', start);
$('colorBy').addEventListener('change', applyColor);
const selection = setupPointSelection({ renderer, renderWindow,
  api: fsrw.getApiSpecificRenderWindow(), actor, view: $('view'), server: SERVER, log,
  getState: () => ({ manifest, positions, recordIds, regionIds, received, complete, epoch }) });
refreshDatasets();
