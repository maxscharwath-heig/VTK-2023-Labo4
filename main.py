import math

import numpy as np
import pyproj
import vtk
from pyproj import Transformer
from vtkmodules.vtkCommonColor import vtkNamedColors

import constants as c

rt90 = pyproj.CRS("EPSG:3021")  # RT90
wgs84 = pyproj.CRS("EPSG:4326")  # LatLon with WGS84 datum
ecef = pyproj.CRS("EPSG:4978")  # ECEF coordinates
rt90_to_wgs84 = Transformer.from_crs(rt90, wgs84)
wgs84_to_ecef = Transformer.from_crs(wgs84, ecef)


def to_vtk_point(altitude, latitude, longitude):
    """
    Convert latitude, longitude, altitude to a point in the VTK coordinate system.
    """
    coordinate_transform = vtk.vtkTransform()

    # Rotate the coordinate transform and translate it to the correct altitude
    coordinate_transform.RotateY(longitude)
    coordinate_transform.RotateX(-latitude)
    coordinate_transform.Translate(0, 0, c.EARTH_RADIUS + altitude)

    return coordinate_transform.TransformPoint(0, 0, 0)


def quadrilateral_interpolation_factors(bounds):
    """
    Calculate the factors for quadrilateral interpolation.
    """
    interpolation_matrix = np.array(
        [[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 0, 1], [1, -1, 1, -1]])

    # Matrix multiplication to get alphas and betas for quadrilateral interpolation
    interpolation_alphas = interpolation_matrix.dot(bounds[:, 0])
    interpolation_betas = interpolation_matrix.dot(bounds[:, 1])

    return interpolation_alphas, interpolation_betas


def quadrilateral_interpolation(x, y, a, b):
    """
    Perform quadrilateral interpolation.
    """
    aa = a[3] * b[2] - a[2] * b[3]

    bb = a[3] * b[0] - a[0] * b[3] + a[1] * \
         b[2] - a[2] * b[1] + x * b[3] - y * a[3]

    cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]

    # Solve quadratic equation
    det = math.sqrt(bb ** 2 - 4 * aa * cc)
    m = (-bb - det) / (2 * aa)
    l = (x - a[0] - a[2] * m) / (a[1] + a[3] * m)

    return l, m


def bounding_box(coords):
    """
    Find the bounding box of a set of coordinates.
    """
    smallest_latitude = coords[:, 0].min()  # Bottom
    biggest_latitude = coords[:, 0].max()  # Top
    smallest_longitude = coords[:, 1].min()  # Left
    biggest_longitude = coords[:, 1].max()  # Right

    return smallest_latitude, biggest_latitude, smallest_longitude, biggest_longitude


def extract_terrain_data():
    altitudes = np.fromfile(
        c.BIL_FILENAME, dtype=np.int16).reshape((c.GRID_SIZE, c.GRID_SIZE))

    latitudes_vector = np.linspace(c.LAT_MAX, c.LAT_MIN, c.GRID_SIZE)
    longitudes_vector = np.linspace(c.LON_MIN, c.LON_MAX, c.GRID_SIZE)

    return altitudes, latitudes_vector, longitudes_vector


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


def extract_glider_data(filename):
    """
    Extract data from the glider path file.
    :param filename: Name of the file with glider data.
    :return: An array with the list x y z of each measure.
    """
    with open(filename) as file:
        file.readline()  # First line is not useful for us.

        # Array that stores each position read.
        coordinates = []

        # Read each measure and extract coordinates
        for line in file.readlines():
            values = line.split()
            coordinates.append(
                (int(values[1]), int(values[2]), float(values[3])))

    return coordinates


def create_map_actor():
    wsg84_corners = np.array([rt90_to_wgs84.transform(x, y)
                              for y, x in c.BOUNDING_COORDS])

    alphas, betas = quadrilateral_interpolation_factors(wsg84_corners)

    # Limits to get the bounding box of the map area to display.
    smallest_latitude, biggest_latitude, smallest_longitude, biggest_longitude = bounding_box(
        wsg84_corners)

    altitudes, latitudes, longitudes = extract_terrain_data()

    points = vtk.vtkPoints()
    altitude_points = vtk.vtkIntArray()

    # Texture coordinates of the points in the bounding box of the area to display
    texture_coordinates = vtk.vtkFloatArray()
    texture_coordinates.SetNumberOfComponents(2)

    bounding_coords_grid = [
        (int((lat - c.LAT_MIN) / c.LAT_STEP),
         int((lon - c.LON_MIN) / c.LON_STEP))
        for lat, lon in wsg84_corners
    ]

    grid_lat_min = min([lat for lat, lon in bounding_coords_grid])
    grid_lat_max = max([lat for lat, lon in bounding_coords_grid])
    grid_lon_min = min([lon for lat, lon in bounding_coords_grid])
    grid_lon_max = max([lon for lat, lon in bounding_coords_grid])

    for i, row in enumerate(altitudes):
        if smallest_latitude <= latitudes[i] <= biggest_latitude:
            # For each longitude, is it in the bounding box of the area to display?
            for j, altitude in enumerate(row):
                if smallest_longitude <= longitudes[j] <= biggest_longitude:
                    # At this point, the lat, long pair is inside the bounding box of the area to display,
                    # so we add it to the structured grid.
                    points.InsertNextPoint(to_vtk_point(
                        altitude, latitudes[i], longitudes[j]))

                    altitude_points.InsertNextValue(altitude)

                    l, m = quadrilateral_interpolation(latitudes[i],
                                                       longitudes[j],
                                                       alphas,
                                                       betas)

                    texture_coordinates.InsertNextTuple((l, m))

    # Preparing structured grid to display the area
    terrain_grid = vtk.vtkStructuredGrid()
    terrain_grid.SetDimensions(
        grid_lon_max - grid_lon_min - 1, grid_lat_max - grid_lat_min, 1)
    terrain_grid.SetPoints(points)
    terrain_grid.GetPointData().SetScalars(altitude_points)
    terrain_grid.GetPointData().SetTCoords(texture_coordinates)

    # Cut the terrain with a plane
    terrain_implicit_boolean = vtk.vtkImplicitBoolean()
    terrain_implicit_boolean.SetOperationTypeToUnion()
    terrain_implicit_boolean.AddFunction(
        clipping_plane(wsg84_corners[0], wsg84_corners[1]))
    terrain_implicit_boolean.AddFunction(
        clipping_plane(wsg84_corners[1], wsg84_corners[2]))
    terrain_implicit_boolean.AddFunction(
        clipping_plane(wsg84_corners[2], wsg84_corners[3]))
    terrain_implicit_boolean.AddFunction(
        clipping_plane(wsg84_corners[3], wsg84_corners[0]))

    # Clipped terrain
    terrain_clipped = vtk.vtkClipDataSet()
    terrain_clipped.SetInputData(terrain_grid)
    terrain_clipped.SetClipFunction(terrain_implicit_boolean)
    terrain_clipped.Update()

    terrain_mapper = vtk.vtkDataSetMapper()
    terrain_mapper.SetInputConnection(terrain_clipped.GetOutputPort())
    terrain_mapper.ScalarVisibilityOff()

    # Loading texture
    jpeg_reader = vtk.vtkJPEGReader()
    jpeg_reader.SetFileName("glider_map.jpg")
    terrain_texture = vtk.vtkTexture()
    terrain_texture.SetInputConnection(jpeg_reader.GetOutputPort())

    # Terrain actor
    terrain_actor = vtk.vtkActor()
    terrain_actor.SetMapper(terrain_mapper)
    terrain_actor.SetTexture(terrain_texture)

    return terrain_actor


def make_glider_path_actor():
    """
    This function creates the glider gps path actor
    :return: Return the corresponding vtkActor
    """

    # Retrieving coordinates from file.
    coords = extract_glider_data("vtkgps.txt")

    # Points that will be used to form the "tube".
    path_points = vtk.vtkPoints()

    # Array to store the altitude delta between points.
    # That allows to color the "tube" accordingly to the difference with the
    # previous point.
    delta_altitudes = vtk.vtkFloatArray()

    last_elev = coords[0][2]

    for i, (x, y, altitude) in enumerate(coords):
        lat, long = rt90_to_wgs84.transform(y, x)  # Convert from rt90 to wsg84.

        # Insert point position.
        path_points.InsertNextPoint(to_vtk_point(altitude, lat, long))
        # Insert point altitude delta
        delta_altitudes.InsertNextValue(last_elev - altitude)

        last_elev = altitude

    # Making lines out of the measures
    path_lines = vtk.vtkLineSource()
    path_lines.SetPoints(path_points)
    path_lines.Update()

    # Setting altitudes.
    path_lines.GetOutput().GetPointData().SetScalars(delta_altitudes)

    # Making the lines thicker with tube filter.
    tube = vtk.vtkTubeFilter()
    tube.SetRadius(25)
    tube.SetInputConnection(path_lines.GetOutputPort())

    # Mapping and actor creation.
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.SetScalarRange((-5, 5))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def make_altitude_text_actor():
    """
    Creates the text actor that will display the intersected altitude.
    We set a white background to make it easy to read even on top of the map.
    :return: A text actor.
    """
    altitude_actor = vtk.vtkTextActor()
    altitude_actor.GetTextProperty().SetColor(0, 0, 0)
    altitude_actor.GetTextProperty().SetBackgroundColor(1, 1, 1)
    altitude_actor.GetTextProperty().SetBackgroundOpacity(1)
    altitude_actor.SetInput("Yolo")
    altitude_actor.GetTextProperty().SetFontSize(20)
    altitude_actor.SetPosition((40, 40))

    return altitude_actor


def make_altitude_strip(terrain_actor):
    sphere = vtk.vtkSphere()
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(sphere)
    cutter.SetInputData(terrain_actor.GetMapper().GetInput())

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())

    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputConnection(stripper.GetOutputPort())
    tube_filter.SetRadius(40)

    altitude_strip_mapper = vtk.vtkDataSetMapper()
    altitude_strip_mapper.SetInputConnection(tube_filter.GetOutputPort())

    altitude_strip_actor = vtk.vtkActor()
    altitude_strip_actor.SetMapper(altitude_strip_mapper)

    return altitude_strip_actor, tube_filter, sphere


def main():
    colors = vtkNamedColors()
    terrain_actor = create_map_actor()
    glider_path_actor = make_glider_path_actor()
    altitude_text_actor = make_altitude_text_actor()
    altitude_strip_actor, tube_filter, sphere = make_altitude_strip(terrain_actor)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(terrain_actor)
    renderer.AddActor(glider_path_actor)
    renderer.AddActor(altitude_text_actor)
    renderer.SetBackground(colors.GetColor3d("White"))

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add an interactor style with a callback to handle the event
    style = vtk.vtkInteractorStyleTrackballCamera()
    render_window_interactor.SetInteractorStyle(style)

    # Create a point picker and add the terrain_actor to the pick list
    picker = vtk.vtkPointPicker()
    picker.PickFromListOn()
    picker.AddPickList(terrain_actor)

    renderer.AddActor(altitude_strip_actor)

    def update_altitude_text(obj, event):
        click_pos = obj.GetInteractor().GetEventPosition()

        # Pick the point on the terrain_actor under the mouse position
        picker.Pick(click_pos[0], click_pos[1], 0, renderer)

        # Get the picked actor
        picked_actor = picker.GetActor()

        # If the picked actor is the terrain
        if picked_actor:
            # Retrieve the altitude of the picked point
            altitude = picker.GetDataSet().GetPointData().GetScalars().GetValue(picker.GetPointId())

            # Update the text actor
            altitude_text_actor.SetInput(f"Altitude: {altitude}m")

            # Update the sphere position and radius
            sphere.SetRadius(altitude + c.EARTH_RADIUS)

            # Update the altitude strip
            tube_filter.Update()

            # Render the updated text
            render_window.Render()

        obj.OnMouseMove()

    style.AddObserver("MouseMoveEvent", update_altitude_text)

    render_window_interactor.Initialize()
    render_window.Render()
    render_window_interactor.Start()


if __name__ == '__main__':
    main()
