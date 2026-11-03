import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkCubeSource from '@kitware/vtk.js/Filters/Sources/CubeSource';
import vtkPointPicker from '@kitware/vtk.js/Rendering/Core/PointPicker';

// Select a rendered point, retain its actual record ID, and ask the backend
// for its region. No nearest-point guess from mouse/world coordinates.
export function setupPointSelection({ renderer, renderWindow, api, actor, view, server, getState, log }) {
  const button = document.getElementById('selectPoint');
  const info = document.getElementById('selectionInfo');
  let armed = false, operation = 0;
  const highlight = vtkPolyData.newInstance();
  const highlightMapper = vtkMapper.newInstance({ scalarVisibility: false });
  highlightMapper.setInputData(highlight);
  const highlightActor = vtkActor.newInstance({ pickable: false });
  highlightActor.setMapper(highlightMapper);
  highlightActor.getProperty().setColor(1, 0.9, 0.05);
  highlightActor.getProperty().setPointSize(4);
  highlightActor.setVisibility(false);
  renderer.addActor(highlightActor);

  const cube = vtkCubeSource.newInstance();
  const cubeMapper = vtkMapper.newInstance({ scalarVisibility: false });
  cubeMapper.setInputConnection(cube.getOutputPort());
  const cubeActor = vtkActor.newInstance({ pickable: false });
  cubeActor.setMapper(cubeMapper);
  cubeActor.getProperty().setRepresentationToWireframe();
  cubeActor.getProperty().setColor(1, 0.9, 0.05);
  cubeActor.getProperty().setLineWidth(2);
  cubeActor.setVisibility(false);
  renderer.addActor(cubeActor);

  const picked = vtkPolyData.newInstance();
  const pickedMapper = vtkMapper.newInstance({ scalarVisibility: false });
  pickedMapper.setInputData(picked);
  const pickedActor = vtkActor.newInstance({ pickable: false });
  pickedActor.setMapper(pickedMapper);
  pickedActor.getProperty().setColor(1, 0.1, 0.8);
  pickedActor.getProperty().setPointSize(12);
  pickedActor.setVisibility(false);
  renderer.addActor(pickedActor);

  function reset() {
    operation++;
    armed = false;
    view.style.cursor = '';
    button.textContent = 'Select point';
    button.disabled = !getState().complete;
    highlightActor.setVisibility(false);
    cubeActor.setVisibility(false);
    pickedActor.setVisibility(false);
    highlight.getPoints().setData(new Float32Array(0), 3);
    highlight.getVerts().setData(new Uint32Array(0));
    info.textContent = 'No point selected.';
    renderWindow.render();
  }

  async function pickPixel(event) {
    const canvas = api.getCanvas();
    const rect = canvas.getBoundingClientRect();
    const [width, height] = api.getSize();
    const x = Math.max(0, Math.min(width - 1, Math.floor((event.clientX - rect.left) * width / rect.width)));
    const y = Math.max(0, Math.min(height - 1, height - 1 - Math.floor((event.clientY - rect.top) * height / rect.height)));
    try {
      const selector = api.getSelector();
      if (!selector) throw new Error('Hardware selection unavailable');
      selector.setFieldAssociation(0); // FIELD_ASSOCIATION_POINTS
      const source = await selector.getSourceDataAsync(renderer,
        Math.max(0, x - 3), Math.max(0, y - 3), Math.min(width - 1, x + 3), Math.min(height - 1, y + 3));
      if (!source) throw new Error('Hardware selection failed');
      // Prefer the exact pixel, then allow a small click tolerance.
      for (let radius = 0; radius <= 3; radius++) {
        const nodes = source.generateSelection(Math.max(0, x - radius), Math.max(0, y - radius),
          Math.min(width - 1, x + radius), Math.min(height - 1, y + radius));
        const node = nodes.find(n => n.getProperties().prop === actor);
        if (node) return node.getSelectionList()[0] ?? -1;
      }
      return -1;
    } catch (error) {
      console.warn('Using CPU point picker:', error.message);
      const picker = vtkPointPicker.newInstance();
      try {
        picker.setPickFromList(true);
        picker.initializePickList();
        picker.addPickList(actor);
        picker.setTolerance(0.003);
        picker.pick([x, y, 0], renderer);
        return picker.getPointId();
      } finally { picker.delete(); }
    } finally { renderWindow.render(); }
  }

  async function selectIndex(index, epoch, token) {
    const state = getState();
    if (state.epoch !== epoch || index < 0 || index >= state.received) return;
    // Display index is arrival order; record ID is dataset order.
    const recordId = state.recordIds[index];
    const response = await fetch(`${server}/point-region?dataset=${encodeURIComponent(state.manifest.dataset_id)}&record_id=${recordId}`);
    if (!response.ok) throw new Error(`Region lookup failed (${response.status})`);
    const result = await response.json();
    if (operation !== token || getState().epoch !== epoch) return;
    if (result.position.some((v, axis) => v !== state.positions[index * 3 + axis])) throw new Error('Dataset changed on the server. Restart the stream.');
    if (result.region_id !== state.regionIds[index]) throw new Error('Server/client region mapping mismatch');

    const wanted = result.region_id;
    let count = 0;
    // Yield between blocks so highlighting a 14M-point dataset does not use
    // one uninterrupted main-thread scan. Only this region gets copied.
    for (let start = 0; start < state.received; start += 100_000) {
      for (let j = start; j < Math.min(start + 100_000, state.received); j++) {
        if (state.regionIds[j] === wanted) count++;
      }
      await new Promise(resolve => setTimeout(resolve, 0));
      if (operation !== token || getState().epoch !== epoch) return;
    }
    const xyz = new Float32Array(count * 3), cells = new Uint32Array(count * 2);
    let out = 0;
    for (let start = 0; start < state.received; start += 100_000) {
      for (let j = start; j < Math.min(start + 100_000, state.received); j++) {
        if (state.regionIds[j] !== wanted) continue;
        xyz.set(state.positions.subarray(j * 3, j * 3 + 3), out * 3);
        cells[out * 2] = 1; cells[out * 2 + 1] = out; out++;
      }
      await new Promise(resolve => setTimeout(resolve, 0));
      if (operation !== token || getState().epoch !== epoch) return;
    }
    highlight.getPoints().setData(xyz, 3);
    highlight.getVerts().setData(cells);
    highlight.modified();
    highlightActor.getProperty().setPointSize(actor.getProperty().getPointSize() + 2);
    highlightActor.setVisibility(true);
    const b = result.bounds;
    cube.setCenter((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2);
    cube.setXLength(b[1] - b[0]); cube.setYLength(b[3] - b[2]); cube.setZLength(b[5] - b[4]);
    cubeActor.setVisibility(true);
    picked.getPoints().setData(new Float32Array(result.position), 3);
    picked.getVerts().setData(new Uint32Array([1, 0]));
    picked.modified(); pickedActor.setVisibility(true);
    info.textContent = `Region R${wanted}\nGrid box: [${result.grid_index.join(', ')}]\n` +
      `Original point ID: ${result.original_point_id}\nID source: ${result.id_kind}\n` +
      `Dataset record ID: ${recordId}\nPosition: ${result.position.map(v => v.toFixed(3)).join(', ')}\n` +
      `Region points: ${count.toLocaleString()} received / ${result.total_points.toLocaleString()} in loaded dataset\n` +
      `Yellow: region points and box. Pink: selected point.`;
    log('point_region_selected', { record_id: recordId, original_point_id: result.original_point_id,
      region_id: wanted, region_received: count, region_total: result.total_points });
    renderWindow.render();
  }

  button.onclick = () => {
    if (!getState().complete) return;
    armed = true; view.style.cursor = 'crosshair';
    button.textContent = 'Click a visible point';
    info.textContent = 'Click a point in the model. Normal dragging resumes afterwards.';
  };
  document.getElementById('clearSelection').onclick = reset;
  view.addEventListener('pointerdown', async event => {
    if (!armed || event.button !== 0) return;
    event.preventDefault(); event.stopImmediatePropagation();
    armed = false; view.style.cursor = '';
    const token = ++operation, epoch = getState().epoch;
    button.disabled = true; info.textContent = 'Selecting point and highlighting its spatial region...';
    try {
      const index = await pickPixel(event);
      if (operation !== token || getState().epoch !== epoch) return;
      if (index < 0) { info.textContent = 'No point hit. Select point again and click directly on the model.'; return; }
      await selectIndex(index, epoch, token);
    } catch (error) {
      if (operation === token) info.textContent = error.message;
      console.error(error);
    } finally {
      if (operation === token) { button.disabled = !getState().complete; button.textContent = 'Select point'; }
    }
  }, true);
  return { reset, ready: () => { button.disabled = false; } };
}
