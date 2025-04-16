import googlemaps

# Initialize with your API key
gmaps = googlemaps.Client(key='AIzaSyDViF_T0eCkBiPz2e9fQyfK0sG8V4WkXiA')

# Coordinates of the property (latitude, longitude)
location = (12.9716, 77.5946)  # Example: Bangalore

# Search for nearby schools within 1 km radius
places_result = gmaps.places_nearby(location=location, radius=1000, type='school')

# Extract place names and distances
for place in places_result['results']:
    print(f"Name: {place['name']}")
    print(f"Address: {place['vicinity']}")
