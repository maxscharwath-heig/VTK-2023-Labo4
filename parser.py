from pyproj import CRS, Transformer

# Initialize coordinate transformation
crs_rt90 = CRS("EPSG:2400")  # RT90
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
