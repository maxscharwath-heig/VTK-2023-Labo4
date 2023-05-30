from tqdm import tqdm
import numpy as np
import pyproj
import vtk

GRID_SIZE = 6000
dem_filename = "EarthEnv-DEM90_N60E010.bil"
des_filename = "DEM90.vtk"

dem_data = np.fromfile(dem_filename, dtype=np.int16)
dem_data = dem_data.reshape((GRID_SIZE, GRID_SIZE)).astype(float)

wgs84 = pyproj.CRS("EPSG:4326")  # LatLon with WGS84 datum
utm = pyproj.CRS("EPSG:32633")  # UTM coordinates

project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

latitude_start = 60
latitude_end = 65
longitude_start = 5
longitude_end = 10

lat_step = (latitude_end - latitude_start) / GRID_SIZE
lon_step = (longitude_end - longitude_start) / GRID_SIZE

bounding_coords = [
    (8.672884552833715, 60.79103974072171),
    (8.814207271697969, 60.815339449115214),
    (8.832192925809254, 60.709118845984214),
    (8.690603591508864, 60.68481654100474)
]

# Calculate the bounding indices
bounding_coords_grid = [(int((lat - latitude_start) / lat_step), int((lon - longitude_start) / lon_step))
                        for lon, lat in bounding_coords]

# Get min and max indices
grid_x_min = min(coord[0] for coord in bounding_coords_grid)
grid_x_max = max(coord[0] for coord in bounding_coords_grid)
grid_y_min = min(coord[1] for coord in bounding_coords_grid)
grid_y_max = max(coord[1] for coord in bounding_coords_grid)

print("Bounding indices: ", grid_x_min, grid_x_max, grid_y_min, grid_y_max)

def process_chunk():
    points = []
    scalars = []

    for x in tqdm(range(grid_x_min, grid_x_max), desc="Processing data", unit="rows"):
        for y in range(grid_y_min, grid_y_max):
            lat = latitude_start + x * lat_step
            lon = longitude_start + y * lon_step
            cord_x, cord_y, cord_z = project(lon, lat, dem_data[x, y])
            points.append((cord_x, cord_y, cord_z))
            scalars.append(cord_z)

    return points, scalars


if __name__ == '__main__':
    points, scalars = process_chunk()

    vtk_points = vtk.vtkPoints()
    vtk_scalars = vtk.vtkFloatArray()

    for point in points:
        vtk_points.InsertNextPoint(point)
    for scalar in scalars:
        vtk_scalars.InsertNextValue(scalar)

    vtk_structured_grid = vtk.vtkStructuredGrid()
    vtk_structured_grid.SetDimensions(grid_y_max - grid_y_min, grid_x_max - grid_x_min, 1)
    vtk_structured_grid.SetPoints(vtk_points)
    vtk_structured_grid.GetPointData().SetScalars(vtk_scalars)
    vtk_structured_grid.Modified()

    print("Writing file: " + des_filename)
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(des_filename)
    writer.SetInputData(vtk_structured_grid)
    writer.Write()

    print("Done, Have a nice day! :)")
