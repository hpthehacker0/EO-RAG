import os
import zipfile
import requests
from dotenv import load_dotenv

load_dotenv()

USERNAME   = os.getenv("CDSE_USER")
PASSWORD   = os.getenv("CDSE_PASSWORD")
PRODUCT_ID = "04e851bf-2ae8-47ca-978c-89a1b554e786"
BANDS      = ["_B04.", "_B08."] # Red and NIR at 10m resolution

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/bands", exist_ok=True)

def get_token():
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    r = requests.post(url, data={
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
    })
    r.raise_for_status()
    return r.json()["access_token"]

def download_zip(token):
    zip_path = "data/raw/product.zip"
    if os.path.exists(zip_path):
        print("Zip already exists, skipping download.")
        return zip_path

    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({PRODUCT_ID})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    print("Resolving redirect...")
    # Step 1: get redirect URL without following it
    r = requests.get(url, headers=headers, allow_redirects=False)
    if r.status_code in (301, 302, 303, 307, 308):
        download_url = r.headers["Location"]
        print(f"Redirected to: {download_url}")
    else:
        download_url = url

    print("Downloading product zip (this will take a few minutes)...")
    # Step 2: download from resolved URL with token
    with requests.get(download_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total_mb = int(r.headers.get("content-length", 0)) / (1024 * 1024)
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                print(f"\r  {downloaded/(1024*1024):.1f} / {total_mb:.1f} MB", end="", flush=True)
    print(f"\nDownload complete: {zip_path}")
    return zip_path

def extract_bands(zip_path):
    print("\nScanning zip for band files...")
    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = z.namelist()
        band_files = [f for f in all_files if any(b in f for b in BANDS) and f.endswith(".jp2")]

        if not band_files:
            print("Band files not found. Listing all .jp2 files for inspection:")
            jp2_files = [f for f in all_files if f.endswith(".jp2")]
            for f in jp2_files[:20]:
                print(f"  {f}")
            return

        print(f"Found {len(band_files)} band files:")
        for bf in band_files:
            print(f"  {bf}")
            filename = os.path.basename(bf)
            out_path = f"data/bands/{filename}"
            with z.open(bf) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            print(f"  -> Extracted to {out_path}")

def main():
    print("Authenticating...")
    token = get_token()
    print("Token OK\n")

    zip_path = download_zip(token)
    extract_bands(zip_path)

    print("\nDone. Band files in data/bands/")
    print("Run: ls data/bands/")

if __name__ == "__main__":
    main()
