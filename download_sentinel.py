import os
import requests
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("CDSE_USER")
PASSWORD = os.getenv("CDSE_PASSWORD")

# Tamil Nadu - Madurai region bounding box
# [min_lon, min_lat, max_lon, max_lat]
BBOX = [77.5, 9.5, 78.5, 10.5]
DATE_START = "2024-01-01"
DATE_END   = "2024-03-31"
MAX_CLOUD  = 30  # % cloud cover filter

def get_token():
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

def search_products(token):
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {MAX_CLOUD}) and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
            f"{BBOX[0]} {BBOX[1]},{BBOX[2]} {BBOX[1]},"
            f"{BBOX[2]} {BBOX[3]},{BBOX[0]} {BBOX[3]},"
            f"{BBOX[0]} {BBOX[1]}))') and "
            f"ContentDate/Start gt {DATE_START}T00:00:00.000Z and "
            f"ContentDate/Start lt {DATE_END}T00:00:00.000Z"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": 3,
    }
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    return r.json()["value"]

def main():
    print("Authenticating...")
    token = get_token()
    print("Token acquired.")

    print(f"Searching Sentinel-2 products over Madurai region...")
    products = search_products(token)

    if not products:
        print("No products found. Try adjusting date or cloud cover filter.")
        return

    print(f"\nFound {len(products)} products:\n")
    for p in products:
        print(f"  ID   : {p['Id']}")
        print(f"  Name : {p['Name']}")
        print(f"  Date : {p['ContentDate']['Start']}")
        print()

if __name__ == "__main__":
    main()
