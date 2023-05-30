import numpy as np
import pyproj
import vtk
from pyproj import Transformer
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor

GRID_SIZE = 6000
dem_filename = "EarthEnv-DEM90_N60E010.bil"

dem_data = np.fromfile(dem_filename, dtype=np.int16)
dem_data = dem_data.reshape((GRID_SIZE, GRID_SIZE)).astype(float)

rt90 = pyproj.CRS("EPSG:3021")  # RT90
wgs84 = pyproj.CRS("EPSG:4326")  # LatLon with WGS84 datum
ecef = pyproj.CRS("EPSG:4978")  # ECEF coordinates

wgs84_to_ecef = Transformer.from_crs(wgs84, ecef)
rt90_to_wgs84 = Transformer.from_crs(rt90, wgs84)

latitude_start = 60
longitude_start = 10

lat_step = lon_step = 5 / GRID_SIZE

bounding_coords = [  # RT90
    (1349340, 7022573),
    (1371573, 7022967),
    (1371835, 7006362),
    (1349602, 7005969)
]

bounding_coords_wgs84 = [rt90_to_wgs84.transform(x, y) for y, x in bounding_coords]
print("Bounding coordinates: ", bounding_coords_wgs84)

# Calculate the bounding indices (convert to wgs84 first)
bounding_coords_grid = [
    (int((lat - latitude_start) / lat_step), int((lon - longitude_start) / lon_step))
    for lat, lon in bounding_coords_wgs84
]

print("Bounding indices: ", bounding_coords_grid)

# Get min and max indices
grid_lat_min = min(coord[0] for coord in bounding_coords_grid)
grid_lat_max = max(coord[0] for coord in bounding_coords_grid)
grid_lon_min = min(coord[1] for coord in bounding_coords_grid)
grid_lon_max = max(coord[1] for coord in bounding_coords_grid)


def process():
    points = vtk.vtkPoints()

    for x in range(grid_lat_min, grid_lat_max):
        for y in range(grid_lon_min, grid_lon_max):
            lat = latitude_start + x * lat_step
            lon = longitude_start + y * lon_step
            z = dem_data[y, x]  # Switching indexing order here
            cord_x, cord_y, cord_z = wgs84_to_ecef.transform(lat, lon, z)
            points.InsertNextPoint((cord_x, cord_y, cord_z))

    vtk_structured_grid = vtk.vtkStructuredGrid()
    vtk_structured_grid.SetDimensions(grid_lon_max - grid_lon_min, grid_lat_max - grid_lat_min, 1)
    vtk_structured_grid.SetPoints(points)
    vtk_structured_grid.Modified()

    # Convert structured grid to polydata
    polydata = grid_to_polydata(vtk_structured_grid)

    return polydata



def grid_to_polydata(structured_grid):
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(structured_grid)
    surf_filter.Update()
    return surf_filter.GetOutput()


def add_texture_coordinates(polydata):
    transformer = Transformer.from_crs(rt90, ecef)
    utm_coords = [transformer.transform(x, y) for x, y in bounding_coords]
    u_min, u_max = min(x for x, y in utm_coords), max(x for x, y in utm_coords)
    v_min, v_max = min(y for x, y in utm_coords), max(y for x, y in utm_coords)

    texture_coords = vtk.vtkFloatArray()
    texture_coords.SetNumberOfComponents(2)
    texture_coords.SetName("Texture Coordinates")
    for i in range(polydata.GetNumberOfPoints()):
        point = polydata.GetPoint(i)
        x, y, _ = point

        # Quad interpolation
        u = (x - u_min) / (u_max - u_min)
        v = (y - v_min) / (v_max - v_min)
        texture_coords.InsertNextTuple([u, v])
    polydata.GetPointData().SetTCoords(texture_coords)


def main():
    # Visualization
    colors = vtkNamedColors()

    # Convert structured grid to polydata
    polydata = process()

    # Add texture coordinates to polydata
    add_texture_coordinates(polydata)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a texture map
    texture_map = vtk.vtkTexture()
    image_reader = vtk.vtkJPEGReader()
    image_reader.SetFileName("glider_map.jpg")
    image_reader.Update()
    texture_map.SetInputDataObject(image_reader.GetOutput())

    # Apply the texture map to the actor
    actor.SetTexture(texture_map)

    # Set up the renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("White"))

    # Set up the render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName('VTK - Labo 4')

    # Set up the interactor
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.Initialize()
    interactor.Start()


if __name__ == '__main__':
    main()
