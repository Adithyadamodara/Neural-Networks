import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
from io import BytesIO
import requests

# Setup Chrome options
chrome_options = webdriver.ChromeOptions()
#chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920x1080")
driver = webdriver.Chrome(options=chrome_options)

# URL to scrape
BASE_URL = "https://www.99acres.com/search/property/buy/delhi?city=1075722"

PAGES_TO_SCRAPE = 5
all_properties = []

for page in range(1, PAGES_TO_SCRAPE + 1):
    print(f"\nScraping page {page}...")

    driver.get(BASE_URL + f"&page={page}")

    try:
        # Wait until listings load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.srpTuple__tupleDetails"))
        )
    except:
        print("❌ Listings not loaded, skipping page.")
        continue

    listings = driver.find_elements(By.CSS_SELECTOR, "div.srpTuple__tupleDetails")

    for listing in listings:
        try:
            title = listing.find_element(By.CSS_SELECTOR, "a.srpTuple__propertyName").text
            location = listing.find_element(By.CSS_SELECTOR, "div.srpTuple__propertyAddress").text

            price = listing.find_element(By.CSS_SELECTOR, "td.srpTuple__midGrid.price").text
            area = listing.find_element(By.CSS_SELECTOR, "td.srpTuple__midGrid.area").text
            bhk = listing.find_element(By.CSS_SELECTOR, "td.srpTuple__midGrid.bedroom").text

            # Get image (if available)
            try:
                img_tag = listing.find_element(By.XPATH, "../../preceding-sibling::div//img")
                img_url = img_tag.get_attribute("src")
            except:
                img_url = None

            width = height = None
            if img_url:
                try:
                    img_resp = requests.get(img_url, timeout=5)
                    if img_resp.ok:
                        image = Image.open(BytesIO(img_resp.content))
                        width, height = image.size
                except:
                    pass

            all_properties.append({
                "Title": title,
                "Location": location,
                "Price": price,
                "Area": area,
                "BHK": bhk,
                "Image_URL": img_url,
                "Image_Width": width,
                "Image_Height": height
            })

        except Exception as e:
            print("Skipped a listing due to error:", e)
            continue

    time.sleep(2)

# Save to CSV
df = pd.DataFrame(all_properties)
df.to_csv("99acres_property_dataset.csv", index=False)
print("\n✅ Data saved to 99acres_property_dataset.csv")

driver.quit()
