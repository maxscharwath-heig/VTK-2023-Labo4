import numpy as np
import pyproj
import vtk
from pyproj import Transformer
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor

GRID_SIZE = 6000
dem_filename = "EarthEnv-DEM90_N60E010.bil"

rt90 = pyproj.CRS("EPSG:3021")  # RT90
wgs84 = pyproj.CRS("EPSG:4326")  # LatLon with WGS84 datum
ecef = pyproj.CRS("EPSG:4978")  # ECEF coordinates

wgs84_to_ecef = Transformer.from_crs(wgs84, ecef)
rt90_to_wgs84 = Transformer.from_crs(rt90, wgs84)

latitude_start = 60
latitude_end = 65
longitude_start = 10
longitude_end = 15

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

def clipping_plane(wgs84_1, wgs84_2):
    """
    Compute a plane that passes through (0,0,0), wgs84_1 and wgs84_2.
    :param wgs84_1: WGS84 coordinate number 1 to pass through.
    :param wgs84_2: WGS84 coordinate number 2 to pass through.
    :return: Return the resulting vtkPlane.
    """

    # Get x,y,z positions of the two additional points.
    p1 = np.array(to_vtk_point(10, wgs84_1[0], wgs84_1[1]))
    p2 = np.array(to_vtk_point(10, wgs84_2[0], wgs84_2[1]))

    # Compute normal for plane orientation
    n = np.cross(p1, p2)

    # Plane creation
    plane = vtk.vtkPlane()
    plane.SetNormal(n)

    return plane

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

def to_vtk_point(altitude, latitude, longitude):
    """
    Convert a geographical coordinate into a vtk point.
    :param altitude: Point's altitude
    :param latitude: Point's latitude
    :param longitude: Point's longitude
    :return: Return the corresponding vtk point.
    """

    coordinate_transform = vtk.vtkTransform()
    coordinate_transform.RotateY(longitude)
    coordinate_transform.RotateX(-latitude)
    coordinate_transform.Translate(0, 0, 6371000 + altitude)

    return coordinate_transform.TransformPoint(0, 0, 0)


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

    wsg84_corners = np.array([
        rt90_to_wgs84.transform(bounding_coords[0][0], bounding_coords[0][1]),
        rt90_to_wgs84.transform(bounding_coords[1][0], bounding_coords[1][1]),
        rt90_to_wgs84.transform(bounding_coords[2][0], bounding_coords[2][1]),
        rt90_to_wgs84.transform(bounding_coords[3][0], bounding_coords[3][1])
    ])

    terrain_grid = vtk.vtkStructuredGrid()
    terrain_grid.SetDimensions(GRID_SIZE, GRID_SIZE, 1)
    # terrain_grid.SetPoints(xyz_points)
    # terrain_grid.GetPointData().SetScalars(altitude_points)
    terrain_grid.GetPointData().SetTCoords(texture_coordinates)


    terrain_implicit_boolean = vtk.vtkImplicitBoolean()
    terrain_implicit_boolean.SetOperationTypeToUnion()
    terrain_implicit_boolean.AddFunction(clipping_plane(wsg84_corners[0], wsg84_corners[1]))
    terrain_implicit_boolean.AddFunction(clipping_plane(wsg84_corners[1], wsg84_corners[2]))
    terrain_implicit_boolean.AddFunction(clipping_plane(wsg84_corners[2], wsg84_corners[3]))
    terrain_implicit_boolean.AddFunction(clipping_plane(wsg84_corners[3], wsg84_corners[0]))

    terrain_clipped = vtk.vtkClipDataSet()
    terrain_clipped.SetInputData(terrain_grid)
    terrain_clipped.SetClipFunction(terrain_implicit_boolean)
    terrain_clipped.Update()

    terrain_mapper = vtk.vtkDataSetMapper()
    terrain_mapper.SetInputConnection(terrain_clipped.GetOutputPort())
    terrain_mapper.ScalarVisibilityOff()

    # Create a texture map
    jpeg_reader = vtk.vtkJPEGReader()
    jpeg_reader.SetFileName("glider_map.jpg")
    terrain_texture = vtk.vtkTexture()
    terrain_texture.SetInputConnection(jpeg_reader.GetOutputPort())

    terrain_actor = vtk.vtkActor()
    terrain_actor.SetMapper(terrain_mapper)
    terrain_actor.SetTexture(terrain_texture)

    # Set up the renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(terrain_actor)
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
