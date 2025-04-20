import pandas as pd
import requests
import time

# Load your dataset
df = pd.read_csv("clean_melb_data.csv")

# Your Google Places API key
API_KEY = 'AIzaSyDViF_T0eCkBiPz2e9fQyfK0sG8V4WkXiA'  

# Define the POI types you want to extract
poi_types = ['school', 'hospital', 'supermarket', 'transit_station']

# Function to get count of POIs within a radius
def get_poi_count(lat, lng, poi_type, radius=1000):
    try:
        url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"location={lat},{lng}&radius={radius}&type={poi_type}&key={API_KEY}"
        )
        response = requests.get(url)
        results = response.json().get("results", [])
        return len(results)
    except:
        return 0

# Initialize columns
for poi in poi_types:
    df[f'num_{poi}s_nearby'] = 0

# Loop through rows and fetch POI counts (rate-limited)
for idx, row in df.iterrows():
    lat = row['Lattitude']
    lng = row['Longtitude']
    for poi in poi_types:
        count = get_poi_count(lat, lng, poi_type=poi)
        df.at[idx, f'num_{poi}s_nearby'] = count
        time.sleep(0.1)  # Avoid hitting rate limits

    # Optional: Save every 100 rows as a checkpoint
    if idx % 100 == 0:
        print(f"Processed {idx} rows...")
        df.to_csv("melb_with_poi_features_partial.csv", index=False)

# Save the final enriched dataset
df.to_csv("melb_with_poi_features.csv", index=False)
print("POI feature extraction complete and saved.")
