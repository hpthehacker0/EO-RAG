import numpy as np
import rasterio
from rasterio.windows import Window
import json
import os
from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)

B04_PATH = "data/bands/T43PHM_20240328T050651_B04.jp2"  # Red
B08_PATH = "data/bands/T43PHM_20240328T050651_B08.jp2"  # NIR

TILE_SIZE = 512   # pixels per tile
DATE      = "2024-03-28"
TILE_NAME = "T43PHM"

os.makedirs("data/ndvi", exist_ok=True)

def compute_ndvi_tiles():
    results = []

    with rasterio.open(B04_PATH) as red_src, rasterio.open(B08_PATH) as nir_src:
        width  = red_src.width
        height = red_src.height
        transform = red_src.transform
        crs = str(red_src.crs)

        print(f"Image size: {width} x {height} pixels")
        print(f"CRS: {crs}")

        tile_count = 0
        for row_off in range(0, height, TILE_SIZE):
            for col_off in range(0, width, TILE_SIZE):
                win = Window(col_off, row_off,
                             min(TILE_SIZE, width  - col_off),
                             min(TILE_SIZE, height - row_off))

                red = red_src.read(1, window=win).astype(float)
                nir = nir_src.read(1, window=win).astype(float)

                # Avoid division by zero
                denom = nir + red
                denom[denom == 0] = np.nan
                ndvi = (nir - red) / denom

                # Skip tiles that are mostly NoData
                valid = ndvi[~np.isnan(ndvi)]
                if len(valid) < (TILE_SIZE * TILE_SIZE * 0.1):
                    continue

                # Geo coordinates of tile center
                tile_transform = rasterio.windows.transform(win, transform)
                center_x = tile_transform.c + (win.width  / 2) * tile_transform.a
                center_y = tile_transform.f + (win.height / 2) * tile_transform.e

                stats = {
                    "tile_id"      : f"{TILE_NAME}_{tile_count:04d}",
                    "date"         : DATE,
                    "col_off"      : col_off,
                    "row_off"      : row_off,
		    "center_lon"   : round(transformer.transform(center_x, center_y)[0], 6),
                    "center_lat"   : round(transformer.transform(center_x, center_y)[1], 6),
                    "ndvi_mean"    : round(float(np.nanmean(ndvi)), 4),
                    "ndvi_min"     : round(float(np.nanmin(ndvi)),  4),
                    "ndvi_max"     : round(float(np.nanmax(ndvi)),  4),
                    "ndvi_std"     : round(float(np.nanstd(ndvi)),  4),
                    "valid_pixels" : int(len(valid)),
                    "vegetation_class": classify_ndvi(float(np.nanmean(ndvi)))
                }

                results.append(stats)
                tile_count += 1

    print(f"\nProcessed {tile_count} valid tiles")
    return results

def classify_ndvi(mean_ndvi):
    if mean_ndvi < 0.0:
        return "water_or_cloud"
    elif mean_ndvi < 0.2:
        return "bare_soil_or_urban"
    elif mean_ndvi < 0.4:
        return "sparse_vegetation"
    elif mean_ndvi < 0.6:
        return "moderate_vegetation"
    else:
        return "dense_vegetation"

def main():
    print("Computing NDVI tiles...\n")
    tiles = compute_ndvi_tiles()

    out_path = "data/ndvi/ndvi_tiles.json"
    with open(out_path, "w") as f:
        json.dump(tiles, f, indent=2)

    print(f"\nSaved {len(tiles)} tiles to {out_path}")

    # Quick summary
    classes = {}
    for t in tiles:
        c = t["vegetation_class"]
        classes[c] = classes.get(c, 0) + 1

    print("\nVegetation class distribution:")
    for cls, count in sorted(classes.items(), key=lambda x: -x[1]):
        print(f"  {cls:30s}: {count} tiles")

    print(f"\nSample tile:")
    print(json.dumps(tiles[0], indent=2))

if __name__ == "__main__":
    main()
