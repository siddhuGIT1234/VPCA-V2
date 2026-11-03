"""Generate sample scientific VTK files to try in the app.

Each dataset has a SMALL but IMPORTANT feature (a "rare group") that random
progressive delivery tends to show late. They are good test cases for
AAHR / VPCA later.

Usage:  python make_samples.py          -> writes ../sample_data/*.vt?
"""
from pathlib import Path

import numpy as np
import pyvista as pv

OUT = Path(__file__).resolve().parent.parent / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)


# --------------------------------------------------------------------------
# 1. Pipe network (.vtu) — the scenario from the original project plan.
#    A main pipe with branches. ONE small side branch has unusually high
#    pressure (a blockage). It's a tiny fraction of the points.
# --------------------------------------------------------------------------
def pipe_network():
    pieces = []

    def pipe(p0, p1, radius, pressure0, pressure1, speed, temp, blocked=False):
        n_len = int(np.linalg.norm(np.subtract(p1, p0)) * 60)
        line = pv.Line(p0, p1, resolution=max(n_len, 10))
        tube = line.tube(radius=radius, n_sides=48, capping=False)
        tube.clear_data()   # drop the tube filter's helper arrays (normals, texture coords)
        t = np.linalg.norm(tube.points - np.array(p0), axis=1) / np.linalg.norm(np.subtract(p1, p0))
        t = np.clip(t, 0, 1)
        d = (np.array(p1) - np.array(p0)) / np.linalg.norm(np.subtract(p1, p0))
        pressure = pressure0 + (pressure1 - pressure0) * t + rng.normal(0, 0.5, tube.n_points)
        # Faster in the middle of the pipe cross-section is not visible on the
        # surface, so add a gentle along-pipe variation instead.
        vel = np.outer(speed * (1 + 0.1 * np.sin(8 * np.pi * t)), d)
        temperature = temp + 5 * t + rng.normal(0, 0.3, tube.n_points)
        if blocked:
            pressure += 80 * np.exp(-((t - 0.7) ** 2) / 0.01)   # sharp pressure spike
            vel *= 0.2
        tube.point_data["pressure"] = pressure
        tube.point_data["velocity"] = vel
        tube.point_data["temperature"] = temperature
        tube.point_data["blocked"] = np.full(tube.n_points, int(blocked))
        pieces.append(tube.cast_to_unstructured_grid())

    pipe((0, 0, 0), (12, 0, 0), 0.6, 200, 120, 3.0, 60)               # main line
    for x in (2, 5, 8, 10.5):                                          # side branches
        pipe((x, 0, 0), (x, 4, 0), 0.25, 170 - 7 * x, 100 - 5 * x, 1.5, 58)
        pipe((x, 0, 0), (x, -3, 1), 0.2, 165 - 7 * x, 105 - 5 * x, 1.2, 57)
    pipe((6.5, 0, 0), (6.5, 1.5, 3), 0.12, 150, 140, 0.8, 75, blocked=True)  # the rare one

    grid = pieces[0].merge(pieces[1:])
    grid.save(OUT / "pipe_network_flow.vtu")
    return "pipe_network_flow.vtu", grid.n_points, "one thin branch with a pressure spike + hot fluid (blocked=1)"


# --------------------------------------------------------------------------
# 2. Vortex flow on a regular grid (.vti) — classic CFD test field
#    (Taylor–Green-like vortices) plus one small hot plume.
# --------------------------------------------------------------------------
def vortex_grid(n=72):
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(2 * np.pi / (n - 1),) * 3)
    x, y, z = grid.points.T
    u = np.sin(x) * np.cos(y) * np.cos(z)
    v = -np.cos(x) * np.sin(y) * np.cos(z)
    w = 0.2 * np.sin(z) * np.cos(x)
    pressure = (np.cos(2 * x) + np.cos(2 * y)) * (np.cos(2 * z) + 2) / 16
    vel = np.stack([u, v, w], axis=1)
    # vorticity magnitude by finite differences on the grid
    U = vel.reshape(n, n, n, 3, order="F")
    h = grid.spacing[0]
    du = [np.gradient(U[..., i], h) for i in range(3)]   # du[i][j] = d(vel_i)/d(axis_j)
    wx = du[2][1] - du[1][2]
    wy = du[0][2] - du[2][0]
    wz = du[1][0] - du[0][1]
    vort = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2).ravel(order="F")
    c = np.array([4.5, 1.8, 3.2])
    plume = np.exp(-np.sum((grid.points - c) ** 2, axis=1) / 0.65)      # small hot spot
    grid.point_data["velocity"] = vel
    grid.point_data["pressure"] = pressure
    grid.point_data["vorticity"] = vort
    grid.point_data["temperature"] = 300 + 2 * pressure + 400 * plume
    grid.save(OUT / "vortex_flow.vti")
    return "vortex_flow.vti", grid.n_points, "a small hot plume (temperature > 500) inside the vortices"


# --------------------------------------------------------------------------
# 3. Galaxy particle simulation (.vtp point cloud) — like astronomy /
#    cosmology particle data. 0.5% of particles are hot star-forming clumps.
# --------------------------------------------------------------------------
def galaxy(n=400_000):
    n_bulge, n_hot = int(n * 0.15), int(n * 0.005)
    n_disk = n - n_bulge - n_hot

    r = rng.exponential(3.0, n_disk) + 0.3
    arm = rng.integers(0, 2, n_disk)
    theta = 0.9 * np.log(r) * 2.5 + arm * np.pi + rng.normal(0, 0.35, n_disk)
    disk = np.stack([r * np.cos(theta), r * np.sin(theta), rng.normal(0, 0.15, n_disk)], 1)

    bulge = rng.normal(0, 0.8, (n_bulge, 3)) * [1, 1, 0.6]

    # hot clumps: a few compact knots placed along the arms
    knots = []
    for k in range(12):
        rk = rng.uniform(2, 9)
        tk = 0.9 * np.log(rk) * 2.5 + (k % 2) * np.pi
        knots.append([rk * np.cos(tk), rk * np.sin(tk), 0])
    knots = np.array(knots)
    hot = knots[rng.integers(0, len(knots), n_hot)] + rng.normal(0, 0.12, (n_hot, 3))

    pos = np.vstack([disk, bulge, hot])
    rr = np.linalg.norm(pos[:, :2], axis=1) + 1e-3
    v_circ = 2.2 * np.tanh(rr / 1.5)
    vel = np.stack([-pos[:, 1] / rr * v_circ, pos[:, 0] / rr * v_circ, rng.normal(0, 0.05, len(pos))], 1)
    temperature = np.concatenate([
        rng.normal(8_000, 1_500, n_disk),
        rng.normal(5_000, 800, n_bulge),
        rng.normal(30_000, 4_000, n_hot),      # the rare hot group
    ])
    density = np.concatenate([
        np.exp(-r / 3) + rng.normal(0, 0.02, n_disk),
        1.5 + rng.normal(0, 0.1, n_bulge),
        3.0 + rng.normal(0, 0.3, n_hot),
    ])
    component = np.concatenate([np.zeros(n_disk), np.ones(n_bulge), np.full(n_hot, 2)]).astype(np.int32)

    cloud = pv.PolyData(pos.astype(np.float32))
    cloud.point_data["temperature"] = temperature
    cloud.point_data["density"] = density
    cloud.point_data["velocity"] = vel
    cloud.point_data["component"] = component   # 0 disk, 1 bulge, 2 hot clumps
    cloud.save(OUT / "galaxy_particles.vtp")
    return "galaxy_particles.vtp", cloud.n_points, "0.5% hot star-forming clumps (component=2, temperature ~30,000)"


# --------------------------------------------------------------------------
# 4. Hurricane-like atmospheric volume (.vtu) — swirling wind, low-pressure
#    eye, warm core, and rare intense updraft cells in the eyewall.
# --------------------------------------------------------------------------
def hurricane(n=350_000):
    r = np.sqrt(rng.uniform(0, 1, n)) * 10
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, 3, n)
    pos = np.stack([r * np.cos(th), r * np.sin(th), z], 1)
    rmax = 2.0
    speed = np.where(r < rmax, 60 * r / rmax, 60 * (rmax / np.maximum(r, 1e-3)) ** 0.6)
    wind = np.stack([-np.sin(th) * speed, np.cos(th) * speed, np.zeros(n)], 1)
    pressure = 1010 - 60 * np.exp(-(r / 3) ** 2) + rng.normal(0, 1, n)
    temperature = 290 - 6 * z + 8 * np.exp(-(r / 1.5) ** 2) + rng.normal(0, 0.5, n)
    updraft = 0.5 * np.exp(-((r - rmax) ** 2) / 0.5) + rng.normal(0, 0.1, n)
    # rare intense convective cells (≈0.3% of points) in the eyewall
    cells_th = rng.uniform(0, 2 * np.pi, 5)
    for ct in cells_th:
        d2 = (r - rmax) ** 2 + (np.angle(np.exp(1j * (th - ct))) * rmax) ** 2 + (z - 1.8) ** 2
        updraft += 25 * np.exp(-d2 / 0.27)
    wind[:, 2] = updraft
    cloud = pv.PolyData(pos.astype(np.float32))
    cloud.point_data["wind"] = wind
    cloud.point_data["pressure"] = pressure
    cloud.point_data["temperature"] = temperature
    grid = cloud.cast_to_unstructured_grid()
    grid.save(OUT / "hurricane.vtu")
    return "hurricane.vtu", grid.n_points, "5 tiny intense updraft cells (wind_z > 10) in the eyewall"


# --------------------------------------------------------------------------
# 5. Terrain surface (legacy .vtk, StructuredGrid) — elevation + slope +
#    rainfall; a narrow river channel is the small feature.
# --------------------------------------------------------------------------
def terrain(n=300):
    x, y = np.meshgrid(np.linspace(0, 20, n), np.linspace(0, 20, n))
    h = (2 * np.sin(x / 3) * np.cos(y / 4) + 1.5 * np.exp(-((x - 12) ** 2 + (y - 8) ** 2) / 10)
         + 0.2 * rng.normal(0, 1, x.shape))
    river = np.exp(-((y - (10 + 3 * np.sin(x / 2.5))) ** 2) / 0.08)
    h -= 1.2 * river
    grid = pv.StructuredGrid(x, y, h)
    gy, gx = np.gradient(h)
    grid.point_data["elevation"] = h.ravel(order="F")
    grid.point_data["slope"] = np.hypot(gx, gy).ravel(order="F")
    grid.point_data["water_depth"] = (1.2 * river).ravel(order="F")
    grid.save(OUT / "terrain.vtk")
    return "terrain.vtk", grid.n_points, "a narrow river channel (water_depth > 0.5)"


if __name__ == "__main__":
    for make in (pipe_network, vortex_grid, galaxy, hurricane, terrain):
        name, npts, rare = make()
        size = (OUT / name).stat().st_size / 1e6
        print(f"{name:24s} {npts:>9,} points  {size:6.1f} MB   rare feature: {rare}")
    print(f"\nWrote files to {OUT}")
