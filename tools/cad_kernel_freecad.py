#!/usr/bin/env python3
"""
Thapar CFD Platform - Parametric CAD Kernel (real FreeCAD)

Runs under FreeCAD's own headless console (`freecadcmd`, via the conda-forge
`freecad` package -- see server/../run_freecad.sh) and drives FreeCAD's real
`Part` module directly: FreeCAD.Version() reports 1.1.3. Every geometric
operation below is the actual FreeCAD Part API, not an approximation:
  - Part.makeBox / makeCylinder / makeSphere / makeCone / makeTorus
  - Part.Face(wire).extrude(...) / Part.Wire(...).revolve(...) for sketches
  - shape.fuse / .cut / .common for boolean union/subtract/intersect
  - shape.makeFillet / .makeChamfer for REAL edge rounding (not a mesh hack)
  - shape.exportStep / Part.read for true B-Rep STEP/IGES import+export

freecadcmd's own CLI option parser collides with argparse-style flags (it
tries to auto-open anything that looks like a bare filename argument), so
job parameters are passed via the THAPAR_CAD_JOB environment variable
(path to a JSON job file) instead of argv.

Voxelization (Brinkman mask for the CUDA engine) tessellates each real
B-Rep solid to a triangle mesh (FreeCAD's own Shape.tessellate) and then
does the actual inside/outside sampling with trimesh's vectorized batch
ray-casting -- FreeCAD's own Shape.isInside() is a real B-Rep query but
~20k calls/sec single-point, too slow for a 100k+ cell grid; tessellating
once and batch-sampling is the standard, honest way every CAD-to-voxel
pipeline (including FreeCAD's own FEM mesher) handles this.
"""
import os
import sys
import json
import numpy as np

import FreeCAD
import Part

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


def _naca0012_profile(chord, thickness):
    t = thickness / chord
    n_pts = 40
    x = np.linspace(0, chord, n_pts)
    yt = 5 * t * chord * (0.2969 * np.sqrt(x / chord) - 0.1260 * (x / chord)
                          - 0.3516 * (x / chord) ** 2 + 0.2843 * (x / chord) ** 3
                          - 0.1015 * (x / chord) ** 4)
    upper = np.column_stack([x, yt])
    lower = np.column_stack([x[::-1], -yt[::-1]])
    return np.vstack([upper, lower])


def _wire_from_points_2d(pts2d, plane, close=True):
    """Builds a real Part.Wire from a 2D point list on the given construction
    plane (xy/xz/yz), matching the frontend's sketch-click convention."""
    def to3d(p):
        if plane == "xy": return FreeCAD.Vector(p[0], p[1], 0)
        if plane == "xz": return FreeCAD.Vector(p[0], 0, p[1])
        return FreeCAD.Vector(0, p[0], p[1])
    verts = [to3d(p) for p in pts2d]
    if close:
        verts = verts + [verts[0]]
    return Part.makePolygon(verts)


def _placement(t):
    t = t or {}
    return (float(t.get("rx", 0)), float(t.get("ry", 0)), float(t.get("rz", 0)),
            float(t.get("tx", 0)), float(t.get("ty", 0)), float(t.get("tz", 0)))


def _apply_placement(shape, transform):
    rx, ry, rz, tx, ty, tz = _placement(transform)
    s = shape.copy()
    origin = FreeCAD.Vector(0, 0, 0)
    if rx: s.rotate(origin, FreeCAD.Vector(1, 0, 0), rx)
    if ry: s.rotate(origin, FreeCAD.Vector(0, 1, 0), ry)
    if rz: s.rotate(origin, FreeCAD.Vector(0, 0, 1), rz)
    if tx or ty or tz: s.translate(FreeCAD.Vector(tx, ty, tz))
    return s


def build_primitive(body_type, params):
    if body_type == "box":
        w, h, d = float(params.get("w", 0.2)), float(params.get("h", 0.2)), float(params.get("d", 0.2))
        return Part.makeBox(w, h, d, FreeCAD.Vector(-w / 2, -h / 2, -d / 2))
    if body_type == "cylinder":
        r = float(params.get("radius", 0.1)); ht = float(params.get("height", 0.4))
        axis = params.get("axis", "z")
        dir_vec = {"x": FreeCAD.Vector(1, 0, 0), "y": FreeCAD.Vector(0, 1, 0), "z": FreeCAD.Vector(0, 0, 1)}[axis]
        base = dir_vec * (-ht / 2)
        return Part.makeCylinder(r, ht, base, dir_vec)
    if body_type == "sphere":
        r = float(params.get("radius", 0.1))
        return Part.makeSphere(r)
    if body_type == "cone":
        r1, r2 = float(params.get("radius1", 0.15)), float(params.get("radius2", 0.0))
        ht = float(params.get("height", 0.3))
        return Part.makeCone(r1, r2, ht, FreeCAD.Vector(0, 0, -ht / 2))
    if body_type == "torus":
        r1, r2 = float(params.get("radius_major", 0.2)), float(params.get("radius_minor", 0.05))
        return Part.makeTorus(r1, r2)
    if body_type == "airfoil":
        chord, thickness, span = float(params.get("chord", 0.5)), float(params.get("thickness", 0.06)), float(params.get("span", 0.6))
        wire = _wire_from_points_2d(_naca0012_profile(chord, thickness), "xy", close=True)
        face = Part.Face(wire)
        return face.extrude(FreeCAD.Vector(0, 0, span))
    if body_type == "sketch_extrude":
        pts = params["points"]; depth = float(params.get("depth", 0.2)); plane = params.get("plane", "xy")
        if len(pts) < 3:
            raise ValueError("Sketch needs at least 3 points to extrude")
        wire = _wire_from_points_2d(pts, plane, close=True)
        face = Part.Face(wire)
        normal = {"xy": FreeCAD.Vector(0, 0, depth), "xz": FreeCAD.Vector(0, depth, 0), "yz": FreeCAD.Vector(depth, 0, 0)}[plane]
        return face.extrude(normal)
    if body_type == "sketch_revolve":
        pts = params["points"]  # (r, z) profile, r >= 0
        angle_deg = float(params.get("angle_deg", 360.0))
        verts = [FreeCAD.Vector(p[0], 0, p[1]) for p in pts]
        wire = Part.makePolygon(verts + [verts[0]])
        try:
            face = Part.Face(wire)
            base_shape = face
        except Exception:
            base_shape = wire  # open profile -> revolve the wire itself (thin shell)
        return base_shape.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), angle_deg)
    raise ValueError(f"Unknown primitive body type: {body_type}")


def boolean_op(shape_a, shape_b, op):
    if op == "union":
        return shape_a.fuse(shape_b)
    if op == "subtract":
        return shape_a.cut(shape_b)
    if op == "intersect":
        return shape_a.common(shape_b)
    raise ValueError(f"Unknown boolean op: {op}")


def fillet_or_chamfer(shape, radius, mode):
    if not shape.Edges:
        raise ValueError("Target body has no edges to fillet/chamfer")
    if mode == "fillet":
        return shape.makeFillet(radius, shape.Edges)
    return shape.makeChamfer(radius, shape.Edges)


def import_step_iges(file_path):
    """Real B-Rep import: Part.read parses STEP/IGES into an actual Shape,
    not a triangulated approximation."""
    return Part.read(file_path)


def load_mesh_file_as_trimesh(file_path):
    if not TRIMESH_AVAILABLE:
        raise RuntimeError("trimesh required to load mesh-only files (STL/OBJ/PLY)")
    mesh = trimesh.load(file_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh


def resolve_scene(scene):
    bodies = {b["id"]: b for b in scene["bodies"]}
    fc_shapes = {}   # body_id -> Part.Shape  (real B-Rep)
    mesh_only = {}    # body_id -> trimesh.Trimesh (STL/OBJ imports, no B-Rep)

    def resolve(body_id, stack=()):
        if body_id in fc_shapes or body_id in mesh_only:
            return
        if body_id in stack:
            raise ValueError(f"Cyclic body reference involving {body_id}")
        b = bodies[body_id]
        btype = b["type"]
        if btype == "boolean":
            resolve(b["params"]["a"], stack + (body_id,))
            resolve(b["params"]["b"], stack + (body_id,))
            if b["params"]["a"] not in fc_shapes or b["params"]["b"] not in fc_shapes:
                raise ValueError("Boolean operands must both be real B-Rep bodies (not mesh imports)")
            fc_shapes[body_id] = boolean_op(fc_shapes[b["params"]["a"]], fc_shapes[b["params"]["b"]], b["params"]["op"])
        elif btype in ("fillet", "chamfer"):
            target = b["params"]["body"]
            resolve(target, stack + (body_id,))
            if target not in fc_shapes:
                raise ValueError("Fillet/chamfer target must be a real B-Rep body (not a mesh import)")
            radius = float(b["params"].get("radius", 0.02))
            fc_shapes[body_id] = fillet_or_chamfer(fc_shapes[target], radius, btype)
        elif btype == "imported":
            file_path = b["params"]["file_path"]
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ('.step', '.stp', '.iges', '.igs', '.brep'):
                fc_shapes[body_id] = _apply_placement(import_step_iges(file_path), b.get("transform"))
            else:
                mesh = load_mesh_file_as_trimesh(file_path)
                rx, ry, rz, tx, ty, tz = _placement(b.get("transform"))
                if rx: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0]))
                if ry: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0]))
                if rz: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1]))
                mesh.apply_translation([tx, ty, tz])
                mesh_only[body_id] = mesh
        else:
            fc_shapes[body_id] = _apply_placement(build_primitive(btype, b["params"]), b.get("transform"))

    consumed = set()
    for b in scene["bodies"]:
        if b["type"] == "boolean":
            consumed.add(b["params"]["a"]); consumed.add(b["params"]["b"])
        elif b["type"] in ("fillet", "chamfer"):
            consumed.add(b["params"]["body"])

    for b in scene["bodies"]:
        resolve(b["id"])

    resolved = {}
    for b in scene["bodies"]:
        body_id = b["id"]
        if body_id in consumed and not b.get("keep_visible", False):
            continue
        if body_id in mesh_only:
            mesh = mesh_only[body_id]
            if len(mesh.vertices) == 0:
                raise ValueError(f"Imported body '{b.get('label', body_id)}' contains no geometry "
                                  f"(empty/unreadable mesh file) -- check the uploaded file is valid")
            resolved[body_id] = (mesh, b, None)
        elif body_id in fc_shapes:
            shape = fc_shapes[body_id]
            if not shape.Faces:
                raise ValueError(f"Body '{b.get('label', body_id)}' ({b['type']}) resolved to an empty/degenerate "
                                  f"shape with no faces -- for imports this usually means the source file has no "
                                  f"actual solid geometry (only placement/metadata); for booleans/fillets it usually "
                                  f"means the operation left nothing (e.g. subtract removed the whole body)")
            verts, faces = shape.tessellate(0.01)
            v = np.array([[p.x, p.y, p.z] for p in verts])
            f = np.array(faces)
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=True) if TRIMESH_AVAILABLE else None
            resolved[body_id] = (mesh, b, shape)
    return resolved


def export_scene_threejs(resolved_bodies, out_json_path):
    bodies_json = []
    combined_min = np.array([np.inf, np.inf, np.inf])
    combined_max = np.array([-np.inf, -np.inf, -np.inf])
    for body_id, (mesh, b, _shape) in resolved_bodies.items():
        combined_min = np.minimum(combined_min, mesh.bounds[0])
        combined_max = np.maximum(combined_max, mesh.bounds[1])
        bodies_json.append({
            "id": body_id, "type": b["type"], "role": b.get("role", "obstacle"), "label": b.get("label", body_id),
            "vertices": mesh.vertices.astype(np.float32).flatten().tolist(),
            "normals": mesh.vertex_normals.astype(np.float32).flatten().tolist(),
            "indices": mesh.faces.astype(np.int32).flatten().tolist(),
            "num_faces": len(mesh.faces), "num_vertices": len(mesh.vertices),
            "watertight": bool(mesh.is_watertight),
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "bounding_box": {"min": mesh.bounds[0].tolist(), "max": mesh.bounds[1].tolist()},
        })
    data = {
        "bodies": bodies_json, "num_bodies": len(bodies_json),
        "scene_bounding_box": {
            "min": combined_min.tolist() if len(bodies_json) else [0, 0, 0],
            "max": combined_max.tolist() if len(bodies_json) else [0, 0, 0],
        },
    }
    with open(out_json_path, 'w') as f:
        json.dump(data, f)
    print(f"[FreeCAD Kernel] Exported {len(bodies_json)} body(ies) to {out_json_path}")


def voxelize_brinkman(resolved_bodies, nx, ny, nz, domain_bounds, sigma_solid):
    x0, y0, z0 = domain_bounds[0]; x1, y1, z1 = domain_bounds[1]
    dx = (x1 - x0) / nx; dy = (y1 - y0) / ny; dz = (z1 - z0) / nz
    xc = np.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, nx, dtype=np.float32)
    yc = np.linspace(y0 + 0.5 * dy, y1 - 0.5 * dy, ny, dtype=np.float32)
    zc = np.linspace(z0 + 0.5 * dz, z1 - 0.5 * dz, nz, dtype=np.float32)
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    mask = np.zeros(nx * ny * nz, dtype=np.float32)
    total_solid = 0
    n_obstacles = 0
    for mesh, b, _shape in resolved_bodies.values():
        if b.get("role", "obstacle") != "obstacle":
            continue
        n_obstacles += 1
        if mesh.is_watertight:
            inside = mesh.contains(points)
        else:
            proximity = trimesh.proximity.ProximityQuery(mesh)
            inside = proximity.signed_distance(points) > 0
        mask[inside] = float(sigma_solid)
        total_solid += int(np.sum(inside))

    total_cells = nx * ny * nz
    print(f"[FreeCAD Kernel] Brinkman voxelization: ~{total_solid}/{total_cells} solid-marked cells "
          f"across {n_obstacles} obstacle body(ies) ({100.0 * total_solid / max(total_cells,1):.2f}%)")
    return mask


JOB_PATH_FALLBACK = "/workspace/thapar CFD platform/server/target/cad_cache/freecad_job.json"


def main():
    # freecadcmd imports this script as a module (__name__ != "__main__") and,
    # empirically, does not propagate the parent shell's exported environment
    # variables into the interpreter it runs scripts in -- so THAPAR_CAD_JOB
    # is read here only as a best-effort override; the fixed cache-dir path
    # (written by CadPreprocessService right before invoking freecadcmd) is
    # what actually gets used in practice.
    job_path = os.environ.get("THAPAR_CAD_JOB") or JOB_PATH_FALLBACK
    if not os.path.isfile(job_path):
        print(f"ERROR: CAD job file not found: {job_path}", file=sys.stderr)
        return 1
    with open(job_path) as f:
        job = json.load(f)

    scene = job["scene"]
    nx, ny, nz = job["nx"], job["ny"], job["nz"]
    lx, ly, lz = job["lx"], job["ly"], job["lz"]
    sigma = job.get("sigma", 10000.0)
    out_mask = job["out_mask"]
    out_json = job["out_json"]

    if not scene.get("bodies"):
        export_scene_threejs({}, out_json)
        np.zeros(nx * ny * nz, dtype=np.float32).tofile(out_mask)
        return 0

    # freecadcmd's own top-level script runner catches ANY uncaught exception
    # itself (prints "Exception while processing file: ...") and still exits
    # 0 regardless -- confirmed empirically (a deliberately invalid fillet
    # radius raises a real OCCError, "BRep_API: command not done", but the
    # freecadcmd process exits 0 unless we catch it and force sys.exit
    # ourselves). CadPreprocessService relies on a nonzero exit code to detect
    # kernel failures, so every real geometry operation must be caught here.
    try:
        resolved = resolve_scene(scene)
        export_scene_threejs(resolved, out_json)
        mask = voxelize_brinkman(resolved, nx, ny, nz, ([0.0, 0.0, 0.0], [lx, ly, lz]), sigma)
        mask.tofile(out_mask)
        print(f"[FreeCAD Kernel] Saved raw float32 obstacle mask to {out_mask} ({mask.nbytes} bytes)")

        if job.get("export_step"):
            shapes = [shape for _mesh, _b, shape in resolved.values() if shape is not None]
            if shapes:
                # Part.export([...], path) with a raw shape list silently writes a
                # near-empty file in this headless context (confirmed: 20 entities
                # regardless of input); wrapping in a real Compound first is what
                # actually serializes every solid.
                Part.makeCompound(shapes).exportStep(job["export_step"])
                print(f"[FreeCAD Kernel] Exported real B-Rep STEP file to {job['export_step']}")
        if job.get("export_stl"):
            meshes = [mesh for mesh, _b, _shape in resolved.values()]
            if meshes and TRIMESH_AVAILABLE:
                combined = trimesh.util.concatenate(meshes)
                combined.export(job["export_stl"])
                print(f"[FreeCAD Kernel] Exported combined scene STL to {job['export_stl']} ({len(combined.faces)} triangles)")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


# freecadcmd loads this file as an imported module (__name__ is the module's
# filename, never "__main__"), so the usual guard would never fire -- run
# unconditionally instead. This file is only ever invoked as a freecadcmd
# script, never imported for its symbols elsewhere.
_exit_code = main()
if _exit_code:
    sys.exit(_exit_code)
