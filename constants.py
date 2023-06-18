# VTK - Labo 4 - Planeur
# Nicolas Crausaz & Maxime Scharwath

GRID_SIZE = 6000

BIL_FILENAME = "EarthEnv-DEM90_N60E010.bil"

# Bounding coordinates of the area (RT90)
BOUNDING_COORDS = [[1349602, 7005969], [1371835, 7006362],
                   [1371573, 7022967], [1349340, 7022573]]

LAT_MIN = 60
LAT_MAX = 65
LON_MIN = 10
LON_MAX = 15
LAT_STEP = (LAT_MAX - LAT_MIN) / GRID_SIZE
LON_STEP = (LON_MAX - LON_MIN) / GRID_SIZE

EARTH_RADIUS = 6371000

# Glider path
G_TUBE_SIZE = 25
G_TUBE_COLORS_RANGE = (-5, 5)
GLIDER_SCALE = (4, 4, 4)
GLIDER_OBJ_PATH = "plane.obj"

# Altitudes interactor
A_TUBE_SIZE = 40