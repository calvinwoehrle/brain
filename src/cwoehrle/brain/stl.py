"""Convert triangle meshes to print-ready STL files.

Calvin Woehrle, 2026
"""

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def export_stl(
    verts: np.ndarray,
    faces: np.ndarray,
    path: Path,
) -> trimesh.Trimesh:
    """Build, repair and validate a triangle mesh, then write a binary STL (mm).

    Marching-cubes surfaces carry small artefacts (degenerate/duplicate faces,
    unreferenced vertices, tiny holes), so we clean and repair the mesh before
    writing a print-ready shell.

    Args:
        verts: ``(N, 3)`` array of vertex coordinates in millimetres.
        faces: ``(M, 3)`` array of triangle vertex indices.
        path: Destination ``.stl`` file path (not a directory).

    Returns:
        The repaired :class:`trimesh.Trimesh` that was written to ``path``.
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Clean up marching-cubes artefacts for a print-ready shell.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    mesh.export(path)  # extension (.stl) selects the exporter
    return mesh


def export_meshes(
    meshes: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path
) -> None:
    """Export a set of named ``(verts, faces)`` meshes to STL files in ``out_dir``.

    Each entry is written to ``out_dir/<name>.stl`` (mm units).
    """
    for name, (verts, faces) in meshes.items():
        out_path = out_dir / f"{name}.stl"
        mesh = export_stl(verts, faces, path=out_path)
        logger.info(
            "Wrote %s: %s triangles, %s vertices, watertight=%s",
            out_path,
            f"{len(mesh.faces):,}",
            f"{len(mesh.vertices):,}",
            mesh.is_watertight,
        )
