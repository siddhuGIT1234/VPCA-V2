"""Run with: python -m pytest test_regions.py"""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as backend
from levels import build_levels
from regions import build_regions, describe_region, point_region_ids


def test_boundaries_and_degenerate_axes():
    p = np.array([[-8, 2, 0], [0, 2, 0], [8, 2, 0]], dtype=np.float32)
    info = build_regions(p)
    assert info['grid'] == [8, 1, 1]
    assert point_region_ids(p, info['minimum'], info['maximum'], info['grid']).tolist() == [0, 4, 7]
    assert sum(info['counts']) == len(p)
    assert describe_region(7, info)['bounds'] == [6, 8, 2, 2, 0, 0]
    same = build_regions(np.ones((3, 3), dtype=np.float32))
    assert same['region_count'] == 1 and same['counts'] == [3]


def test_every_point_belongs_to_its_box_and_counts():
    p = np.random.default_rng(12).normal(size=(20_000, 3)).astype(np.float32)
    info = build_regions(p)
    ids = point_region_ids(p, info['minimum'], info['maximum'], info['grid'])
    assert sum(info['counts']) == len(p)
    assert len(np.unique(ids)) > 100
    for i in range(0, len(p), 77):
        b = np.asarray(describe_region(int(ids[i]), info)['bounds']).reshape(3, 2)
        assert np.all(p[i] >= b[:, 0]) and np.all(p[i] <= b[:, 1])


def test_server_browser_region_parity():
    p = np.random.default_rng(8).uniform(-30, 40, (2000, 3)).astype(np.float32)
    info = build_regions(p)
    # Include internal boundaries, global maxima, and degenerate bounds.
    p = np.vstack([p, info['minimum'], info['maximum'],
                   np.asarray(info['minimum']) + (np.asarray(info['maximum']) - info['minimum']) / 2])
    expected = point_region_ids(p, info['minimum'], info['maximum'], info['grid'])
    module = (Path(__file__).parent.parent / 'client/src/regions.js').as_uri()
    script = f"import {{regionId}} from '{module}'; import fs from 'node:fs'; const d=JSON.parse(fs.readFileSync(0,'utf8')); console.log(JSON.stringify(d.points.map(p=>regionId(p,d.info))));"
    result = subprocess.run(['node', '--input-type=module', '-e', script],
                            input=json.dumps({'points': p.tolist(), 'info': info}),
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == expected.tolist()


@pytest.fixture
def dataset(tmp_path):
    p = np.random.default_rng(5).uniform(-2, 2, (5000, 3)).astype(np.float32)
    # Deliberately different from dataset row IDs and random arrival indexes.
    original = np.arange(5000, dtype=np.float32)[::-1] + 100_000
    data = dict(record_id=np.arange(len(p), dtype=np.int32), pos=p,
                att=np.column_stack([original, p[:, 0]]), label=np.zeros(len(p), dtype=np.int32))
    file = tmp_path / 'test.npz'
    np.savez(file, **data, attribute_names=np.array(['OriginalPointId', 'value']))
    backend._register('test_mapping', file)
    yield data
    backend._datasets.pop('test_mapping', None)
    for key in list(backend._levels):
        if key[0] == 'test_mapping': del backend._levels[key]


def test_api_and_random_stream_identity(dataset):
    with TestClient(backend.app) as client:
        m = client.get('/manifest', params={'dataset': 'test_mapping'}).json()
        assert m['record_count'] == 5000 and sum(m['regions']['counts']) == 5000
        rows = []
        with client.websocket_connect('/stream?dataset=test_mapping&method=random') as ws:
            while True:
                header = ws.receive_json()
                if header['type'] == 'done': break
                payload = ws.receive_bytes()
                assert len(payload) == header['bytes']
                rows.append(np.frombuffer(payload, dtype=np.float32).reshape(-1, m['stride']))
        rows = np.concatenate(rows)
        assert sorted(rows[:, 0].astype(int)) == list(range(5000))
        for row in rows[::103]:
            record = int(row[0])
            r = client.get('/point-region', params={'dataset': 'test_mapping', 'record_id': record}).json()
            assert r['position'] == row[1:4].tolist()
            assert r['original_point_id'] == int(row[4])
            assert r['record_id'] == record
            assert r['total_points'] == m['regions']['counts'][r['region_id']]
        assert client.get('/point-region?dataset=test_mapping&record_id=5000').status_code == 404
        assert client.get('/point-region?dataset=test_mapping&record_id=-1').status_code == 422
        assert client.get('/point-region?dataset=missing&record_id=0').status_code == 404


def test_legacy_dataset_without_original_id(dataset):
    ds = backend._datasets['test_mapping']
    ds['names'] = ['value']
    ds['data']['att'] = ds['data']['att'][:, 1:]
    with TestClient(backend.app) as client:
        result = client.get('/point-region?dataset=test_mapping&record_id=24').json()
        assert result['original_point_id'] == 24
        assert result['id_kind'] == 'dataset record index'


def test_reimport_invalidates_region_cache(dataset, tmp_path):
    old = backend.get_regions('test_mapping')
    dataset['pos'] = dataset['pos'] + 100
    path = tmp_path / 'replacement.npz'
    np.savez(path, **dataset, attribute_names=np.array(['OriginalPointId', 'value']))
    backend._register('test_mapping', path)
    assert backend.get_regions('test_mapping')['minimum'] != old['minimum']
