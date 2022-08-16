# Import modules
import functools
import os
import shutil
import time
from rsgislib.segmentation import shepherdseg
import configparser
import geopandas as gpd
import pandas as pd
import subprocess
from rasterstats import zonal_stats
from osgeo import gdal


def compute_area(shp_path, field_name):
    shape = gpd.read_file(shp_path)
    shape[field_name] = shape["geometry"].area
    shape.to_file(shp_path)


def delete_shp(base_dir, shape_name):
    path = os.path.join(base_dir, shape_name)
    os.remove(path + ".shp")
    os.remove(path + ".cpg")
    os.remove(path + ".dbf")
    os.remove(path + ".prj")
    os.remove(path + ".shx")


def join_shp(first_shp_path, second_shp_path, out_path):
    first_shp = gpd.read_file(first_shp_path)
    second_shp = gpd.read_file(second_shp_path)
    join = gpd.sjoin(first_shp, second_shp, how="inner", predicate="intersects")
    join = join[["id", "geometry", "area_seg", "area_ref"]]
    join["geometry"] = join.buffer(0.01)
    join = join.dissolve(by="id", aggfunc="mean")
    join.to_file(out_path)


def union_shp(first_shp_path, second_shp_path, out_path):
    first_shp = gpd.read_file(first_shp_path)
    second_shp = gpd.read_file(second_shp_path)
    union = gpd.sjoin(first_shp, second_shp, how="inner", predicate="intersects")
    union["area_uni"] = union["geometry"].area
    union["id"] = union["id_left"]
    union = union[["id", "geometry", "area_uni"]]
    union.to_file(out_path)


def intersect_shp(first_shp_path, second_shp_path, out_path):
    first_shp = gpd.read_file(first_shp_path)
    second_shp = gpd.read_file(second_shp_path)
    intersection = gpd.overlay(first_shp, second_shp, how="intersection")
    intersection["area_int"] = intersection["geometry"].area
    intersection["id"] = intersection["id_1"]
    intersection = intersection[["id", "geometry", "area_int"]]
    intersection = intersection.dissolve(by="id", aggfunc="sum")
    intersection.to_file(out_path)


def accuracy(join_shp_path, union_shp_path, intersect_shp_path):
    shp_join = gpd.read_file(join_shp_path)
    shp_union = gpd.read_file(union_shp_path)
    shp_intersect = gpd.read_file(intersect_shp_path)
    accuracy_table = functools.reduce(functools.partial(pd.merge, on="id"), [shp_join, shp_union, shp_intersect])
    accuracy_table = accuracy_table[["id", "area_seg", "area_ref", "area_uni", "area_int"]]
    accuracy_table['QR'] = 1 - (accuracy_table['area_int'] / accuracy_table['area_uni'])
    accuracy_table['AFI'] = (accuracy_table['area_ref'] - accuracy_table['area_seg']) / accuracy_table['area_ref']
    accuracy_table['OS'] = 1 - (accuracy_table['area_int'] / accuracy_table['area_ref'])
    accuracy_table['US'] = 1 - (accuracy_table['area_int'] / accuracy_table['area_seg'])
    accuracy_table['RMSE'] = ((accuracy_table['OS'] ** 2 + accuracy_table['US'] ** 2) / 2) ** 0.5
    rmse = accuracy_table["RMSE"].mean()
    print("RMSE: " + str(rmse))


def compute_stats(shape_path, image_path, output_path):
    print("Computing stats")
    shape = gpd.read_file(shape_path)
    image = gdal.Open(image_path)
    n = 1
    while n <= image.RasterCount:
        shape["mean_" + str(n)] = pd.DataFrame(zonal_stats(vectors=shape["geometry"],
                                                           raster=image_path,
                                                           band=n,
                                                           stats="mean"))
        print(f"Stats for band {n} computed")
        n += 1
    shape.to_file(output_path)


def segmentation_step(month, raster_path, vector_path):
    references = os.path.join(vector_path, f"references_{month}.shp")
    image_tif = os.path.join(raster_path, f"{month}.tif")
    image_kea = os.path.join(raster_path, f"{month}.kea")
    subprocess.call(f"gdal_translate -of KEA {image_tif} {image_kea}", shell=True)
    result_folder = f"result_{str(month)}"
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
        os.mkdir(result_folder)
    else:
        os.mkdir(result_folder)
    if month == "august":
        k = 15
        s = 25
        p = 105
        d = 90
    elif month == "january":
        k = 20
        s = 110
        p = 105
        d = 95
    output_clumps = os.path.join(raster_path, "segmentation.kea")
    if os.path.exists(output_clumps):
        os.remove(output_clumps)
    segmentation_output = os.path.join(result_folder, "segmentation.shp")
    if os.path.exists(segmentation_output):
        os.remove(segmentation_output)
    shepherdseg.run_shepherd_segmentation(input_img=image_tif,
                                          out_clumps_img=output_clumps,
                                          tmp_dir="./tmp",
                                          num_clusters=k,
                                          min_n_pxls=p,
                                          dist_thres=d,
                                          sampling=s,
                                          calc_stats=False)
    subprocess.call(f"gdal_polygonize.py {output_clumps} -f 'ESRI Shapefile' {segmentation_output}", shell=True)
    # Segment area
    compute_area(shp_path=segmentation_output,
                 field_name="area_seg")
    # Reference area
    compute_area(shp_path=references,
                 field_name="area_ref")
    # Segment and reference join table
    join_path = os.path.join(result_folder, "area_s_r.shp")
    join_shp(first_shp_path=segmentation_output,
             second_shp_path=references,
             out_path=join_path)
    # Union between segment and reference
    union_path = os.path.join(result_folder, "union.shp")
    union_shp(first_shp_path=join_path,
              second_shp_path=references,
              out_path=union_path)
    # Intersection between segment and reference
    intersection_path = os.path.join(result_folder, "intersection.shp")
    intersect_shp(first_shp_path=join_path,
                  second_shp_path=references,
                  out_path=intersection_path)
    accuracy(join_shp_path=join_path,
             union_shp_path=union_path,
             intersect_shp_path=intersection_path)
    compute_stats(shape_path=segmentation_output,
                  image_path=image_tif,
                  output_path=os.path.join(result_folder, f"{month}_stats.shp"))
    delete_shp(base_dir=result_folder, shape_name="area_s_r")
    delete_shp(base_dir=result_folder, shape_name="union")
    delete_shp(base_dir=result_folder, shape_name="intersection")


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")
    raster_path = config.get("folders", "raster")
    vector_path = config.get("folders", "vector")
    month = config.get("season", "month")
    print(f"The process of image acquired in {month} will start in 10 seconds...")
    time.sleep(10)
    segmentation_step(month=month,
                      raster_path=raster_path,
                      vector_path=vector_path)


if __name__ == "__main__":
    main()
