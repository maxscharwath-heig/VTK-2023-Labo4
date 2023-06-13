import math

import numpy as np
import pyproj
import vtk
from pyproj import Transformer
from vtkmodules.vtkCommonColor import vtkNamedColors

import constants as c

rt90_to_wgs84 = Transformer.from_crs(
    pyproj.CRS("EPSG:3021"), pyproj.CRS("EPSG:4326"))
wgs84_to_ecef = Transformer.from_crs(
    pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:4978"))


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
        [[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 0, 1], [1, -1, 1, -1]]
    )

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


def extract_map_data():
    altitudes = np.fromfile(
        c.BIL_FILENAME, dtype=np.int16).reshape((c.GRID_SIZE, c.GRID_SIZE))

    latitudes_vector = np.linspace(c.LAT_MAX, c.LAT_MIN, c.GRID_SIZE)
    longitudes_vector = np.linspace(c.LON_MIN, c.LON_MAX, c.GRID_SIZE)

    return altitudes, latitudes_vector, longitudes_vector


def clipping_plane(wgs84_1, wgs84_2):
    """
    Create a clipping plane from two points in WGS84 and origin.
    """

    # Get x,y,z positions of the two additional points.
    p1 = np.array(to_vtk_point(0, wgs84_1[0], wgs84_1[1]))
    p2 = np.array(to_vtk_point(0, wgs84_2[0], wgs84_2[1]))

    # Compute normal for plane orientation and create plane.
    n = np.cross(p1, p2)
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
        size = file.readline()
        # Array that stores each position read.
        coordinates = []

        # Read each measure and extract coordinates
        for line in file.readlines():
            values = line.split()
            coordinates.append(
                (int(values[1]), int(values[2]), float(values[3]))
            )

    return coordinates


def create_map_actor():
    wsg84_corners = np.array([rt90_to_wgs84.transform(x, y)
                             for y, x in c.BOUNDING_COORDS])

    alphas, betas = quadrilateral_interpolation_factors(wsg84_corners)

    # Limits to get the bounding box of the map area to display.
    smallest_latitude, biggest_latitude, smallest_longitude, biggest_longitude = bounding_box(
        wsg84_corners)

    altitudes, latitudes, longitudes = extract_map_data()

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
        # Check if the latitude of the row is inside the bounding box
        if smallest_latitude <= latitudes[i] <= biggest_latitude:
            for j, altitude in enumerate(row):
                # Check if the longitude of the column is inside the bounding box
                if smallest_longitude <= longitudes[j] <= biggest_longitude:

                    points.InsertNextPoint(
                        to_vtk_point(
                            altitude, latitudes[i], longitudes[j])
                    )

                    altitude_points.InsertNextValue(altitude)

                    l, m = quadrilateral_interpolation(latitudes[i],
                                                       longitudes[j],
                                                       alphas,
                                                       betas)

                    texture_coordinates.InsertNextTuple((l, m))

    # Preparing structured grid to display the area
    map_grid = vtk.vtkStructuredGrid()
    map_grid.SetDimensions(
        grid_lon_max - grid_lon_min - 1, grid_lat_max - grid_lat_min, 1)
    map_grid.SetPoints(points)
    map_grid.GetPointData().SetScalars(altitude_points)
    map_grid.GetPointData().SetTCoords(texture_coordinates)

    # Cut the map with planes to fit the area
    map_implicit = vtk.vtkImplicitBoolean()
    map_implicit.SetOperationTypeToUnion()

    lenCorners = len(wsg84_corners)
    for i in range(lenCorners):
        map_implicit.AddFunction(
            clipping_plane(wsg84_corners[i], wsg84_corners[(i + 1) % lenCorners]))

    # Clipped map
    map_clipped = vtk.vtkClipDataSet()
    map_clipped.SetInputData(map_grid)
    map_clipped.SetClipFunction(map_implicit)
    map_clipped.Update()

    map_mapper = vtk.vtkDataSetMapper()
    map_mapper.SetInputConnection(map_clipped.GetOutputPort())
    map_mapper.ScalarVisibilityOff()

    # Loading and mapping texture
    jpeg_reader = vtk.vtkJPEGReader()
    jpeg_reader.SetFileName("glider_map.jpg")
    map_texture = vtk.vtkTexture()
    map_texture.SetInputConnection(jpeg_reader.GetOutputPort())

    # map actor
    map_actor = vtk.vtkActor()
    map_actor.SetMapper(map_mapper)
    map_actor.SetTexture(map_texture)

    return map_actor


def make_glider_path_actor():
    # Retrieving coordinates from file
    coords = extract_glider_data("vtkgps.txt")

    # Coordinates of the glider path
    path_points = vtk.vtkPoints()

    # Difference of altitude between two measures (for coloring)
    delta_altitudes = vtk.vtkFloatArray()

    previous_altitude = coords[0][2]  # First altitude of the path

    for (x, y, altitude) in coords:
        latitude, longitude = rt90_to_wgs84.transform(y, x)

        # Insert glider path point
        path_points.InsertNextPoint(
            to_vtk_point(altitude, latitude, longitude))

        # Insert difference of altitude
        delta_altitudes.InsertNextValue(previous_altitude - altitude)
        previous_altitude = altitude

    # Create lines between each point
    path_lines = vtk.vtkLineSource()
    path_lines.SetPoints(path_points)
    path_lines.Update()
    path_lines.GetOutput().GetPointData().SetScalars(delta_altitudes)

    # Creating tubes around the lines
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
    altitude_actor.GetTextProperty().SetBackgroundOpacity(0.7)
    altitude_actor.SetInput("Altitude: -")
    altitude_actor.GetTextProperty().SetFontSize(40)
    altitude_actor.SetPosition((40, 40))

    return altitude_actor


def make_altitude_strip(map_actor):
    sphere = vtk.vtkSphere()
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(sphere)
    cutter.SetInputData(map_actor.GetMapper().GetInput())

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
    map_actor = create_map_actor()
    glider_path_actor = make_glider_path_actor()
    altitude_text_actor = make_altitude_text_actor()
    altitude_strip_actor, tube_filter, sphere = make_altitude_strip(
        map_actor)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(map_actor)
    renderer.AddActor(glider_path_actor)
    renderer.AddActor(altitude_text_actor)
    renderer.SetBackground(colors.GetColor3d("Wheat"))

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add an interactor style with a callback to handle the event
    style = vtk.vtkInteractorStyleTrackballCamera()
    render_window_interactor.SetInteractorStyle(style)

    # Create a point picker and add the map_actor to the pick list
    picker = vtk.vtkPointPicker()
    picker.PickFromListOn()
    picker.AddPickList(map_actor)

    renderer.AddActor(altitude_strip_actor)

    def update_altitude_text(obj, event):
        click_pos = obj.GetInteractor().GetEventPosition()

        # Pick the point on the map_actor under the mouse position
        picker.Pick(click_pos[0], click_pos[1], 0, renderer)

        # Get the picked actor
        picked_actor = picker.GetActor()

        # If the picked actor is the map
        if picked_actor:
            # Retrieve the altitude of the picked point
            altitude = picker.GetDataSet().GetPointData(
            ).GetScalars().GetValue(picker.GetPointId())

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
