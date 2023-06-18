# VTK - Labo 4 - Planeur
# Nicolas Crausaz & Maxime Scharwath

import math
import numpy as np
import pyproj
import vtk
from pyproj import Transformer
from vtkmodules.vtkCommonColor import vtkNamedColors

import constants as c

# Transformer to convert between coordinate systems (RT90 and WGS84)
rt90_to_wgs84 = Transformer.from_crs(
    pyproj.CRS("EPSG:3021"),
    pyproj.CRS("EPSG:4326")
)


def to_cartesian(altitude, latitude, longitude):
    """
    Convert latitude, longitude, altitude to a point in the cartesian coordinate system.
    """

    # Rotate the point to the correct position
    coordinate_transform = vtk.vtkTransform()
    coordinate_transform.RotateY(longitude)
    coordinate_transform.RotateX(-latitude)

    return coordinate_transform.TransformPoint(0, 0, c.EARTH_RADIUS + altitude)


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
    """
    Extract data from the map file.
    """

    altitudes = np.fromfile(
        c.BIL_FILENAME, dtype=np.int16).reshape((c.GRID_SIZE, c.GRID_SIZE))

    latitudes_vector = np.linspace(c.LAT_MAX, c.LAT_MIN, c.GRID_SIZE)
    longitudes_vector = np.linspace(c.LON_MIN, c.LON_MAX, c.GRID_SIZE)

    return altitudes, latitudes_vector, longitudes_vector


def clipping_plane(coord1, coord2):
    """
    Create a clipping plane from two points in WGS84 and origin.
    """

    # Compute normal for plane orientation and create plane.
    plane = vtk.vtkPlane()
    n = np.cross(
        to_cartesian(0, coord1[0], coord1[1]),
        to_cartesian(0, coord2[0], coord2[1])
    )
    plane.SetNormal(n)

    return plane


def extract_glider_data(filename):
    """
    Extract data from the glider data file.
    """
    with open(filename) as file:
        file.readline()

        # List of coordinates
        coordinates = []

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

    # Bounds of the area to display
    smallest_latitude, biggest_latitude, smallest_longitude, biggest_longitude = bounding_box(
        wsg84_corners)

    altitudes, latitudes, longitudes = extract_map_data()

    points = vtk.vtkPoints()
    altitude_points = vtk.vtkIntArray()

    # Texture coordinates for the map
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
                        to_cartesian(
                            altitude, latitudes[i], longitudes[j])
                    )

                    altitude_points.InsertNextValue(altitude)

                    l, m = quadrilateral_interpolation(latitudes[i],
                                                       longitudes[j],
                                                       alphas,
                                                       betas)

                    texture_coordinates.InsertNextTuple((l, m))

    # Structured grid for the map
    map_grid = vtk.vtkStructuredGrid()
    map_grid.SetDimensions(
        grid_lon_max - grid_lon_min - 1, grid_lat_max - grid_lat_min, 1)
    map_grid.SetPoints(points)
    map_grid.GetPointData().SetScalars(altitude_points)
    map_grid.GetPointData().SetTCoords(texture_coordinates)

    # Cut the map with planes to fit the area
    map_implicit = vtk.vtkImplicitBoolean()
    map_implicit.SetOperationTypeToUnion()

    len_corners = len(wsg84_corners)
    for i in range(len_corners):
        map_implicit.AddFunction(
            clipping_plane(wsg84_corners[i], wsg84_corners[(i + 1) % len_corners]))

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

    # Map actor
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
            to_cartesian(altitude, latitude, longitude))

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
    tube.SetRadius(c.G_TUBE_SIZE)
    tube.SetInputConnection(path_lines.GetOutputPort())

    # Mapping and actor creation.
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.SetScalarRange(c.G_TUBE_COLORS_RANGE)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor, coords


def make_altitude_text_actor():
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
    tube_filter.SetRadius(c.A_TUBE_SIZE)

    altitude_strip_mapper = vtk.vtkDataSetMapper()
    altitude_strip_mapper.SetInputConnection(tube_filter.GetOutputPort())

    altitude_strip_actor = vtk.vtkActor()
    altitude_strip_actor.SetMapper(altitude_strip_mapper)

    return altitude_strip_actor, tube_filter, sphere


def create_plane_actor(initial_position):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(c.GLIDER_OBJ_PATH)
    reader.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Set the initial position of the glider
    x, y, altitude = initial_position
    lat, long = rt90_to_wgs84.transform(y, x)
    x, y, z = to_cartesian(altitude, lat, long)
    actor.SetPosition(x, y, z)
    actor.SetScale(c.GLIDER_SCALE)

    return actor


# Create an animation of the glider following the path
class GliderAnimator:
    def __init__(self, glider_model, path):
        self.glider_model = glider_model
        self.path = path
        # Index of current position on the path, need to be float for interpolation
        self.path_position = 1

    def move_glider(self, obj, event):
        # If we've reached the end of the path, reset to the beginning
        if self.path_position >= len(self.path):
            self.path_position = 1

        # Get the next position and altitude from the path
        index = int(self.path_position)
        x1, y1, altitude1 = self.path[index - 1]
        x2, y2, altitude2 = self.path[index]
        altitude = altitude1 + (altitude2 - altitude1) * \
            (self.path_position - index)
        lat, long = rt90_to_wgs84.transform(
            y1 + (y2 - y1) * (self.path_position - index),
            x1 + (x2 - x1) * (self.path_position - index)
        )

        # Compute the position in VTK coordinates
        x, y, z = to_cartesian(altitude, lat, long)

        # Move the glider model to the new position
        self.glider_model.SetPosition(x, y, z)
        z_angle = math.atan2(y2 - y1, x2 - x1)
        self.glider_model.SetOrientation(0, math.degrees(z_angle) + 90, 0)

        # Increment the path position for the next move
        self.path_position += 0.5

        # Render the updated scene
        obj.GetRenderWindow().Render()


def main():
    colors = vtkNamedColors()
    # Actors creation
    map_actor = create_map_actor()
    glider_path_actor, glider_path = make_glider_path_actor()
    altitude_text_actor = make_altitude_text_actor()
    altitude_strip_actor, tube_filter, sphere = make_altitude_strip(
        map_actor)
    glider_actor = create_plane_actor(glider_path[0])

    renderer = vtk.vtkRenderer()
    renderer.AddActor(map_actor)
    renderer.AddActor(glider_path_actor)
    renderer.AddActor(altitude_text_actor)
    renderer.SetBackground(colors.GetColor3d("Wheat"))
    renderer.AddActor(glider_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 1000)

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
        else:
            altitude_text_actor.SetInput("Altitude: -")
            sphere.SetRadius(c.EARTH_RADIUS)
            tube_filter.Update()
            render_window.Render()

        obj.OnMouseMove()

    style.AddObserver("MouseMoveEvent", update_altitude_text)

    # Create a GliderAnimator object
    glider_animator = GliderAnimator(glider_actor, glider_path)

    # Add the glider animator as an observer to the interactor
    render_window_interactor.AddObserver(
        "TimerEvent", glider_animator.move_glider)
    render_window_interactor.CreateRepeatingTimer(33)

    # Center the window on the screen & start
    screen_size = render_window.GetScreenSize()
    window_size = render_window.GetSize()
    render_window.SetPosition(
        int((screen_size[0] - window_size[0]) / 2),
        int((screen_size[1] - window_size[1]) / 2)
    )
    render_window_interactor.Initialize()
    render_window_interactor.Start()


if __name__ == '__main__':
    main()
