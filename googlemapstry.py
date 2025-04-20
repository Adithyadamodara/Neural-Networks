import requests
import time
import pandas as pd
import requests
import time

API_KEY = "AIzaSyDViF_T0eCkBiPz2e9fQyfK0sG8V4WkXiA"
df = pd.read_csv("clean_melb_data.csv")

def get_coordinates(suburb):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    response = requests.get(base_url, params={"address": suburb + ", Australia", "key": API_KEY})
    result = response.json()
    if result["status"] == "OK":
        loc = result["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    else:
        return None, None

# Add coordinates to dataframe
df["Latitude"], df["Longitude"] = zip(*df["Suburb"].apply(get_coordinates))
df.dropna(inplace=True)
df.to_csv("geocoded_melb_data.csv", index=False)

def get_coordinates(suburb):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    for attempt in range(3):  # Retry 3 times
        try:
            response = requests.get(base_url, params={"address": f"{suburb}, Australia", "key": API_KEY}, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data["results"]:
                location = data["results"][0]["geometry"]["location"]
                return location["lat"], location["lng"]
            return None, None
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error for '{suburb}' -> {e}")
            time.sleep(1)
    return None, None


