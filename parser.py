from pyproj import CRS, Transformer

# Initialize coordinate transformation
crs_rt90 = CRS("EPSG:3021")  # RT90
crs_wgs84 = CRS("EPSG:4326")  # WGS84
transformer = Transformer.from_crs(crs_rt90, crs_wgs84)

converted_data = []

# Read the file and convert coordinates in a single loop
with open('vtkgps.txt', 'r') as file:
    for line in file:
        if line.startswith('T'):
            parts = line.split()
            x = int(parts[1])
            y = int(parts[2])
            altitude = float(parts[3])
            date = parts[4]
            time = parts[5]

            # Convert coordinates from RT90 to WGS84
            lon, lat = transformer.transform(x, y)

            converted_data.append((x, y, altitude, date, time, lon, lat))

# Print the converted data
for record in converted_data[:5]:
    print(record)

# Print the converted data ( of the given image )
# Haut-gauche (8.672884552833715, 60.79103974072171)
# Haut-droite (8.814207271697969, 60.815339449115214)
# Bas-droite (8.832192925809254, 60.709118845984214)
# Bas-gauche (8.690603591508864, 60.68481654100474)
coords_rt90 = [
    (1349340, 7022573),  # Haut-gauche (8.672884552833715, 60.79103974072171)
    (1371573, 7022967),  # Haut-droite (8.814207271697969, 60.815339449115214)
    (1371835, 7006362),  # Bas-droite (8.832192925809254, 60.709118845984214)
    (1349602, 7005969)   # Bas-gauche (8.690603591508864, 60.68481654100474)
]

converted_coords = []

for coord in coords_rt90:
    # Convert coordinates from RT90 to WGS84
    lon, lat = transformer.transform(coord[1], coord[0])
    converted_coords.append((lat, lon))

# Print the converted coordinates
for coord in converted_coords:
    print(coord)
