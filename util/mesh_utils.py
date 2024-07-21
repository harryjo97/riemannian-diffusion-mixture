from evtk import hl, vtk
import numpy as np

def trimesh_to_vtk(filename, v, f, *, cell_data={}, point_data={}):
    v = np.array(v)
    f = np.array(f)

    for key, value in cell_data.items():
        cell_data[key] = (
            np.array(value)
        )

    for key, value in point_data.items():
        point_data[key] = (
            np.array(value)
        )

    n_cells = f.shape[0]
    hl.unstructuredGridToVTK(
        path=filename,
        x=v[:, 0].copy(order="F"),
        y=v[:, 1].copy(order="F"),
        z=v[:, 2].copy(order="F"),
        connectivity=f.reshape(-1),
        offsets=np.arange(start=3, stop=3 * (n_cells + 1), step=3, dtype="uint32"),
        cell_types=np.ones(n_cells, dtype="uint8") * vtk.VtkTriangle.tid,
        cellData=cell_data,
        pointData=point_data,
    )


def points_to_vtk(filename, pts):
    pts = np.array(pts)
    hl.pointsToVTK(
        filename,
        x=pts[:, 0].copy(order="F"),
        y=pts[:, 1].copy(order="F"),
        z=pts[:, 2].copy(order="F"),
    )