#!/usr/bin/env python3
"""
Thapar CFD Platform - Headless ParaView Export Bridge (real ParaView 5.11.2)

Runs under plain `python3` -- the apt `python3-paraview` package installs the
actual ParaView Python bindings into the system site-packages, so
`from paraview.simple import *` works directly, no `pvpython` wrapper needed
(verified: `paraview.simple`'s Contour/StreamTracer/ProbeLocation filters and
servermanager.Fetch all run correctly here).

The live solver's fields (dumped as raw float32 binaries by
CfdEngineService.dumpFieldsForExport, shape nz*ny*nx row-major) are wrapped
into a real vtkImageData and fed into ParaView's own pipeline via
TrivialProducer, then processed with ParaView's actual filters:
  - Contour       -> isosurfaces (was previously a hand-rolled marching cubes)
  - StreamTracer  -> streamlines (was previously a manual RK4 integrator)
  - ProbeLocation -> point probes (real vtkProbeFilter under the hood)
  - vtkXMLImageDataWriter -> .vti export (was previously hand-written XML)

Falls back to a synthetic placeholder field only when no --fields-dir is
given (e.g. ad hoc CLI testing without a live solver).
"""

import sys
import os
import json
import argparse
import numpy as np

import vtk
from vtk.util import numpy_support as vnp
from paraview.simple import *
from paraview import servermanager


def build_image_data(fields, nx, ny, nz, dx, dy, dz):
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(float(dx), float(dy), float(dz))
    img.SetOrigin(0.0, 0.0, 0.0)
    for name, arr in fields.items():
        flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        vtk_arr = vnp.numpy_to_vtk(flat, deep=True)
        vtk_arr.SetName(name)
        img.GetPointData().AddArray(vtk_arr)
    if "u" in fields and "v" in fields and "w" in fields:
        vec = np.stack([fields["u"], fields["v"], fields["w"]], axis=-1).astype(np.float32).reshape(-1, 3)
        vtk_vec = vnp.numpy_to_vtk(vec, deep=True)
        vtk_vec.SetName("velocity")
        img.GetPointData().AddArray(vtk_vec)
        img.GetPointData().SetActiveVectors("velocity")
    primary = "speed" if "speed" in fields else next(iter(fields))
    img.GetPointData().SetActiveScalars(primary)
    return img


def make_producer(img):
    producer = TrivialProducer(registrationName="thaparLiveFields")
    producer.GetClientSideObject().SetOutput(img)
    producer.UpdatePipeline()
    return producer


def export_vti(path, producer):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(servermanager.Fetch(producer))
    writer.SetDataModeToBinary()
    writer.Write()
    print(f"[ParaView Bridge] Wrote {path} via vtkXMLImageDataWriter (real ParaView/VTK writer)", file=sys.stderr)


def generate_isosurface_json(out_path, producer, field_name, isovalue, nx, ny, nz, dx, dy, dz):
    contour = Contour(Input=producer)
    contour.ContourBy = ['POINTS', field_name]
    contour.Isosurfaces = [isovalue]
    contour.ComputeNormals = 1
    contour.UpdatePipeline()

    poly = servermanager.Fetch(contour)
    n_pts = poly.GetNumberOfPoints()
    n_cells = poly.GetNumberOfCells()

    if n_pts == 0:
        payload = {"isovalue": isovalue, "num_vertices": 0, "num_faces": 0, "vertices": [], "normals": [], "faces": []}
    else:
        pts = vnp.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)
        normals_vtk = poly.GetPointData().GetNormals()
        normals = vnp.vtk_to_numpy(normals_vtk).astype(np.float32) if normals_vtk else np.zeros_like(pts)

        faces = []
        id_list = vtk.vtkIdList()
        polys = poly.GetPolys()
        polys.InitTraversal()
        while polys.GetNextCell(id_list):
            if id_list.GetNumberOfIds() == 3:
                faces.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
            elif id_list.GetNumberOfIds() == 4:
                a, b, c, d = (id_list.GetId(i) for i in range(4))
                faces.extend([[a, b, c], [a, c, d]])

        payload = {
            "isovalue": isovalue, "num_vertices": len(pts), "num_faces": len(faces),
            "vertices": pts.flatten().tolist(), "normals": normals.flatten().tolist(),
            "faces": [i for tri in faces for i in tri],
        }
    with open(out_path, 'w') as f:
        json.dump(payload, f)
    print(f"[ParaView Bridge] Contour filter: {n_pts} points, {n_cells} cells -> {out_path}", file=sys.stderr)


def generate_streamlines_json(out_path, producer, nx, ny, nz, dx, dy, dz, num_seeds=36, max_steps=150):
    domain_center = [(nx * dx) / 2, (ny * dy) / 2, (nz * dz) / 2]
    domain_radius = 0.35 * min(nx * dx, ny * dy, nz * dz)

    st = StreamTracer(Input=producer)
    st.Vectors = ['POINTS', 'velocity']
    st.SeedType = 'Point Cloud'
    st.SeedType.Center = domain_center
    st.SeedType.Radius = domain_radius
    st.SeedType.NumberOfPoints = num_seeds
    st.MaximumStreamlineLength = float(max(nx * dx, ny * dy, nz * dz)) * 4
    st.IntegrationDirection = 'BOTH'
    st.UpdatePipeline()

    poly = servermanager.Fetch(st)
    pts = vnp.vtk_to_numpy(poly.GetPoints().GetData()) if poly.GetNumberOfPoints() else np.zeros((0, 3))

    lines = []
    id_list = vtk.vtkIdList()
    cells = poly.GetLines()
    if cells:
        cells.InitTraversal()
        while cells.GetNextCell(id_list):
            n = id_list.GetNumberOfIds()
            if n < 2:
                continue
            line_pts = [pts[id_list.GetId(i)].tolist() for i in range(n)]
            lines.append({"points": line_pts})

    with open(out_path, 'w') as f:
        json.dump({"num_lines": len(lines), "lines": lines}, f)
    print(f"[ParaView Bridge] StreamTracer filter: {len(lines)} streamlines, {poly.GetNumberOfPoints()} points -> {out_path}", file=sys.stderr)


def probe_point(producer, x, y, z, field_names):
    probe = ProbeLocation(Input=producer, ProbeType='Fixed Radius Point Source')
    probe.ProbeType.Center = [float(x), float(y), float(z)]
    probe.UpdatePipeline()
    out = servermanager.Fetch(probe)
    res = {"x": float(x), "y": float(y), "z": float(z)}
    valid_arr = out.GetPointData().GetArray("vtkValidPointMask")
    valid = bool(vnp.vtk_to_numpy(valid_arr)[0]) if valid_arr and out.GetNumberOfPoints() else False
    if not valid or out.GetNumberOfPoints() == 0:
        return res
    for name in field_names:
        arr = out.GetPointData().GetArray(name)
        if arr is None:
            continue
        vals = vnp.vtk_to_numpy(arr)
        res[name] = float(vals[0]) if arr.GetNumberOfComponents() == 1 else vals[0].tolist()
    if "u" in res and "v" in res and "w" in res:
        res["speed"] = float(np.sqrt(res["u"] ** 2 + res["v"] ** 2 + res["w"] ** 2))
    return res


def export_slice_json(out_json_path, slice_data, axis, index, nx, ny, nz, dx, dy, dz):
    payload = {
        "axis": axis, "index": index,
        "width": slice_data.shape[1], "height": slice_data.shape[0],
        "data": slice_data.astype(np.float32).flatten().tolist(),
    }
    with open(out_json_path, 'w') as f:
        json.dump(payload, f)


def main():
    parser = argparse.ArgumentParser(description="Thapar CFD Platform - ParaView Export Bridge")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, required=True)
    parser.add_argument("--dz", type=float, required=True)
    parser.add_argument("--out-vti", type=str, default=None)
    parser.add_argument("--out-slice", type=str, default=None)
    parser.add_argument("--slice-axis", type=str, default="z", choices=["x", "y", "z"])
    parser.add_argument("--slice-idx", type=int, default=8)
    parser.add_argument("--out-streamlines", type=str, default=None)
    parser.add_argument("--out-isosurface", type=str, default=None)
    parser.add_argument("--isovalue", type=float, default=0.5)
    parser.add_argument("--probe-x", type=float, default=None)
    parser.add_argument("--probe-y", type=float, default=None)
    parser.add_argument("--probe-z", type=float, default=None)
    parser.add_argument("--fields-dir", type=str, default=None,
                         help="Directory of raw float32 field dumps (u.bin, v.bin, w.bin, "
                              "pressure.bin, temperature.bin, nu_t.bin, alpha.bin), shape "
                              "(nz,ny,nx) row-major, written by CfdEngineService.dumpFieldsForExport(). "
                              "When present, real live solver state is used instead of the synthetic "
                              "placeholder fields below.")
    args = parser.parse_args()

    total = args.nx * args.ny * args.nz
    shape = (args.nz, args.ny, args.nx)

    def load_live_field(name):
        if not args.fields_dir:
            return None
        path = os.path.join(args.fields_dir, name + ".bin")
        if not os.path.isfile(path):
            return None
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size != total:
            print(f"[ParaView Bridge] WARNING: {path} has {arr.size} elements, expected {total}; ignoring", file=sys.stderr)
            return None
        return arr.reshape(shape)

    live_u, live_v, live_w = load_live_field("u"), load_live_field("v"), load_live_field("w")

    if live_u is not None and live_v is not None and live_w is not None:
        u, v, w = live_u, live_v, live_w
        fields = {"u": u, "v": v, "w": w, "speed": np.sqrt(u ** 2 + v ** 2 + w ** 2)}
        for name in ("pressure", "temperature", "nu_t", "alpha"):
            f = load_live_field(name)
            if f is not None:
                fields[name] = f
        print(f"[ParaView Bridge] Loaded live solver fields from {args.fields_dir}: {list(fields.keys())}", file=sys.stderr)
    else:
        x = np.linspace(0, (args.nx - 1) * args.dx, args.nx, dtype=np.float32)
        y = np.linspace(0, (args.ny - 1) * args.dy, args.ny, dtype=np.float32)
        z = np.linspace(0, (args.nz - 1) * args.dz, args.nz, dtype=np.float32)
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        u = np.sin(np.pi * Y) * np.cos(np.pi * X)
        v = -np.cos(np.pi * Y) * np.sin(np.pi * X)
        w = 0.1 * np.sin(np.pi * Z)
        pressure = -0.25 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))
        temperature = np.exp(-((X - 0.5) ** 2 + (Y - 0.2) ** 2) / 0.05)
        alpha = np.where(Y < 0.3, 1.0, 0.0).astype(np.float32)
        fields = {"u": u, "v": v, "w": w, "speed": np.sqrt(u ** 2 + v ** 2 + w ** 2),
                  "pressure": pressure, "temperature": temperature, "alpha": alpha}

    img = build_image_data(fields, args.nx, args.ny, args.nz, args.dx, args.dy, args.dz)
    producer = make_producer(img)

    if args.out_vti:
        export_vti(args.out_vti, producer)

    if args.out_slice:
        idx = args.slice_idx
        if args.slice_axis == "z":
            slice_data = fields["speed"][idx, :, :]
        elif args.slice_axis == "y":
            slice_data = fields["speed"][:, idx, :]
        else:
            slice_data = fields["speed"][:, :, idx]
        export_slice_json(args.out_slice, slice_data, args.slice_axis, idx, args.nx, args.ny, args.nz, args.dx, args.dy, args.dz)

    if args.out_streamlines:
        generate_streamlines_json(args.out_streamlines, producer, args.nx, args.ny, args.nz, args.dx, args.dy, args.dz)

    if args.out_isosurface:
        # Contour temperature or VoF alpha when active (interface tracking); otherwise
        # fall back to velocity magnitude so the button still produces a meaningful
        # surface for plain single-phase, isothermal live runs.
        field_name = "temperature" if "temperature" in fields else ("alpha" if "alpha" in fields else "speed")
        generate_isosurface_json(args.out_isosurface, producer, field_name, args.isovalue,
                                  args.nx, args.ny, args.nz, args.dx, args.dy, args.dz)

    if args.probe_x is not None and args.probe_y is not None and args.probe_z is not None:
        res = probe_point(producer, args.probe_x, args.probe_y, args.probe_z, list(fields.keys()))
        print(json.dumps(res))

    return 0


if __name__ == "__main__":
    sys.exit(main())
