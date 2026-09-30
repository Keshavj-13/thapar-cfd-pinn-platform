#!/usr/bin/env python3
"""
Thapar CFD Platform - Headless CAD Preprocessing & Brinkman Voxelization Bridge
Ingests STEP/IGES/BREP (via OpenCASCADE/Gmsh) and STL/OBJ/PLY (via Trimesh).
Extracts B-Rep face topology for Three.js face-picker and generates Cartesian
Brinkman solid penalization masks sigma(x,y,z) for CUDA engine libthaparcfd.so.
"""

import sys
import os
import json
import argparse
import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


def create_primitive_mesh(primitive_type="cylinder", radius=0.15, length=0.6):
    """Generates standard aerodynamic and hydrodynamic benchmark geometries."""
    if not TRIMESH_AVAILABLE:
        raise RuntimeError("Trimesh required for primitive generation")
        
    if primitive_type == "cylinder":
        # Circular cylinder obstacle aligned with Z axis
        mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=64)
        mesh.apply_translation([0.3, 0.5, 0.5 * length])
    elif primitive_type == "sphere":
        # Sphere obstacle
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        mesh.apply_translation([0.35, 0.5, 0.5])
    elif primitive_type == "airfoil":
        # NACA 0012 symmetric airfoil
        c = 0.5
        t = 0.12
        n_pts = 60
        x = np.linspace(0, c, n_pts)
        yt = 5 * t * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)
        
        upper = np.column_stack([x, yt])
        lower = np.column_stack([x[::-1], -yt[::-1]])
        poly = np.vstack([upper, lower])
        from shapely.geometry import Polygon
        polygon = Polygon(poly)
        mesh = trimesh.creation.extrude_polygon(polygon, height=length)
        mesh.apply_translation([0.25, 0.5, 0.5 * (1.0 - length)])
    else:
        # Box obstacle
        mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        mesh.apply_translation([0.5, 0.5, 0.5])
        
    return mesh


def process_cad_file(filepath, lc=0.05):
    """Processes STEP/IGES via Gmsh OpenCASCADE or STL/OBJ via Trimesh."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.step', '.stp', '.iges', '.igs', '.brep'] and GMSH_AVAILABLE:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cad_geometry")
        
        gmsh.model.occ.importShapes(filepath)
        gmsh.model.occ.synchronize()
        
        # Set mesh size
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.model.mesh.generate(2) # Surface 2D triangulation
        
        # Extract node coordinates
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        vertices = np.array(coords).reshape(-1, 3)
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
        
        # Extract triangular elements per physical face
        faces_list = []
        face_entities = gmsh.model.getEntities(2)
        face_groups = {}
        
        for face_dim, face_tag in face_entities:
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(face_dim, face_tag)
            for etype, enodes in zip(elem_types, elem_node_tags):
                if etype == 2: # 3-node triangle
                    tri_nodes = [tag_to_idx[t] for t in enodes]
                    triangles = np.array(tri_nodes).reshape(-1, 3)
                    start_idx = len(faces_list)
                    faces_list.extend(triangles.tolist())
                    end_idx = len(faces_list)
                    face_groups[f"face_{face_tag}"] = list(range(start_idx, end_idx))
                    
        gmsh.finalize()
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces_list), process=True)
        return mesh, face_groups
        
    elif TRIMESH_AVAILABLE:
        mesh = trimesh.load(filepath)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        # Tag faces using normal clustering
        face_groups = {}
        normals = mesh.face_normals
        # Categorize into cardinal directions
        face_groups["x_pos"] = np.where(normals[:, 0] > 0.7)[0].tolist()
        face_groups["x_neg"] = np.where(normals[:, 0] < -0.7)[0].tolist()
        face_groups["y_pos"] = np.where(normals[:, 1] > 0.7)[0].tolist()
        face_groups["y_neg"] = np.where(normals[:, 1] < -0.7)[0].tolist()
        face_groups["z_pos"] = np.where(normals[:, 2] > 0.7)[0].tolist()
        face_groups["z_neg"] = np.where(normals[:, 2] < -0.7)[0].tolist()
        face_groups["body"] = list(range(len(mesh.faces)))
        return mesh, face_groups
    else:
        raise RuntimeError("No suitable CAD or mesh parser available")


def export_threejs_geometry(mesh, face_groups, out_json_path):
    """Exports indexed mesh geometry and face groups as JSON for Three.js CAD Viewport."""
    vertices = mesh.vertices.astype(np.float32).flatten().tolist()
    normals = mesh.vertex_normals.astype(np.float32).flatten().tolist()
    indices = mesh.faces.astype(np.int32).flatten().tolist()
    
    bbox = {
        "min": mesh.bounds[0].tolist(),
        "max": mesh.bounds[1].tolist(),
        "center": mesh.centroid.tolist(),
        "extents": mesh.extents.tolist()
    }
    
    data = {
        "vertices": vertices,
        "normals": normals,
        "indices": indices,
        "face_groups": face_groups,
        "bounding_box": bbox,
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces)
    }
    
    with open(out_json_path, 'w') as f:
        json.dump(data, f)
    print(f"[CAD Bridge] Exported Three.js CAD geometry to {out_json_path} ({len(mesh.faces)} triangles)")


def voxelize_brinkman(mesh, nx=64, ny=64, nz=64, domain_bounds=([0, 0, 0], [1, 1, 1]), sigma_solid=10000.0):
    """
    Computes 3D Cartesian Brinkman solid penalization mask sigma(x, y, z).
    Points inside the solid are marked with sigma_solid (damping = 1 / (1 + dt*sigma)).
    """
    x0, y0, z0 = domain_bounds[0]
    x1, y1, z1 = domain_bounds[1]
    
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz
    
    xc = np.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, nx, dtype=np.float32)
    yc = np.linspace(y0 + 0.5 * dy, y1 - 0.5 * dy, ny, dtype=np.float32)
    zc = np.linspace(z0 + 0.5 * dz, z1 - 0.5 * dz, nz, dtype=np.float32)
    
    # 3D grid of cell centers (Z, Y, X layout matching C engine)
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate point containment inside watertight solid
    if mesh.is_watertight:
        inside = mesh.contains(points)
    else:
        # Fallback to ray casting or signed distance
        print("[CAD Bridge] Warning: Mesh is not perfectly watertight, using proximity signed distance query")
        proximity = trimesh.proximity.ProximityQuery(mesh)
        distances = proximity.signed_distance(points)
        inside = distances > 0  # Convention: positive inside
        
    mask = np.zeros(nx * ny * nz, dtype=np.float32)
    mask[inside] = float(sigma_solid)
    
    solid_cells = int(np.sum(inside))
    total_cells = nx * ny * nz
    solid_pct = 100.0 * solid_cells / total_cells
    print(f"[CAD Bridge] Brinkman Voxelization: {solid_cells} / {total_cells} cells inside solid ({solid_pct:.2f}%)")
    
    return mask


def main():
    parser = argparse.ArgumentParser(description="Thapar CFD Platform - Headless CAD Preprocessing")
    parser.add_argument("--input", type=str, default=None, help="Path to STEP/STL/OBJ file")
    parser.add_argument("--primitive", type=str, default=None, choices=["cylinder", "sphere", "airfoil", "box"], help="Generate built-in primitive")
    parser.add_argument("--nx", type=int, default=64, help="Domain grid Nx")
    parser.add_argument("--ny", type=int, default=64, help="Domain grid Ny")
    parser.add_argument("--nz", type=int, default=64, help="Domain grid Nz")
    parser.add_argument("--lx", type=float, default=1.0, help="Domain Lx")
    parser.add_argument("--ly", type=float, default=1.0, help="Domain Ly")
    parser.add_argument("--lz", type=float, default=1.0, help="Domain Lz")
    parser.add_argument("--sigma", type=float, default=10000.0, help="Solid Brinkman obstacle damping")
    parser.add_argument("--out-mask", type=str, default="obstacle_mask.bin", help="Output raw float32 binary mask")
    parser.add_argument("--out-json", type=str, default="cad_geometry.json", help="Output Three.js JSON geometry")
    
    args = parser.parse_args()
    
    if args.input:
        print(f"[CAD Bridge] Loading CAD file: {args.input}")
        mesh, face_groups = process_cad_file(args.input)
    elif args.primitive:
        print(f"[CAD Bridge] Generating standard CFD primitive: {args.primitive}")
        mesh = create_primitive_mesh(args.primitive)
        face_groups = {"body": list(range(len(mesh.faces)))}
    else:
        print("[CAD Bridge] Defaulting to circular cylinder obstacle benchmark")
        mesh = create_primitive_mesh("cylinder", radius=0.1, length=0.8)
        face_groups = {"cylinder_body": list(range(len(mesh.faces)))}
        
    # 1. Export Three.js CAD model JSON
    export_threejs_geometry(mesh, face_groups, args.out_json_path if hasattr(args, 'out_json_path') else args.out_json)
    
    # 2. Voxelize to 3D Cartesian Brinkman mask
    domain_bounds = ([0.0, 0.0, 0.0], [args.lx, args.ly, args.lz])
    mask = voxelize_brinkman(mesh, args.nx, args.ny, args.nz, domain_bounds, args.sigma)
    
    # 3. Save raw float32 binary mask for direct ingest by C engine thapar_set_obstacle_mask()
    mask.tofile(args.out_mask)
    print(f"[CAD Bridge] Saved raw float32 obstacle mask to {args.out_mask} ({mask.nbytes} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
