import os
import configparser
import random

import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def index_classification(points):
    class_index = []
    for index, row in points.iterrows():
        if row["Class_name"] == "Lake":
            class_index.append(0)
        if row["Class_name"] == "Dark soil":
            class_index.append(1)
        if row["Class_name"] == "Light soil":
            class_index.append(2)
        if row["Class_name"] == "Vegetated":
            class_index.append(3)
        if row["Class_name"] == "Densely vegetated":
            class_index.append(4)
    points["class_index"] = class_index
    return points


def label_classification(polygons):
    class_name = []
    for index, row in polygons.iterrows():
        if row["class_ind"] == 0:
            class_name.append("Lake")
        if row["class_ind"] == 1:
            class_name.append("Dark soil")
        if row["class_ind"] == 2:
            class_name.append("Light soil")
        if row["class_ind"] == 3:
            class_name.append("Vegetated")
        if row["class_ind"] == 4:
            class_name.append("Densely vegetated")
    polygons["class_name"] = class_name
    return polygons


def prepare_classification_data(stats_path, points_path):
    stats_shp = gpd.read_file(stats_path)
    points_shp = gpd.read_file(points_path)
    points = index_classification(points=points_shp)
    classification_data = gpd.sjoin(stats_shp, points, how="inner", predicate="intersects")
    x_train = classification_data[["mean_1", "mean_2", "mean_3", "mean_4"]]
    y_train = classification_data[["class_index"]]
    return x_train, y_train, points


def equal_random_stratified(month, classified_shp_path, field_name, number_segments, result_folder):
    classified_shp = gpd.read_file(classified_shp_path)
    unique_values = classified_shp[field_name].unique().tolist()
    geometry_df = []
    class_name_df = []
    for value in unique_values:
        geometry = []
        class_name = []
        for index, row in classified_shp.iterrows():
            if row[field_name] == value:
                geometry.append(row["geometry"])
                class_name.append(row[field_name])
        random_index = random.sample(range(1, len(class_name)), number_segments)
        for i in random_index:
            geometry_df.append(geometry[i])
            class_name_df.append(class_name[i])
    data = {
        "geometry": geometry_df,
        "class_name": class_name_df
    }
    gdf = gpd.GeoDataFrame(data).set_crs(epsg=32633)
    gdf.to_file(os.path.join(result_folder, f"{month}_equal_random_stratified.shp"))


def classification_step(x, y, month, stats_path, result_folder):
    stats_shp = gpd.read_file(stats_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(x_train, y_train.values.ravel())
    y_prediction = rfc.predict(x_test)
    print(confusion_matrix(y_test, y_prediction))
    print(classification_report(y_test, y_prediction))
    print(accuracy_score(y_test, y_prediction))
    prediction = stats_shp[['mean_1', 'mean_2', 'mean_3', 'mean_4']]
    stats_shp["class_ind"] = rfc.predict(prediction)
    labeled_shp = label_classification(stats_shp)
    stats_shp = labeled_shp.drop("class_ind", axis=1)
    stats_shp.to_file(os.path.join(result_folder, f"{month}_classification.shp"))


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")
    vector_path = config.get("folders", "vector")
    month = config.get("season", "month")
    result_folder = f"result_{str(month)}"
    shp_name = f"{str(month)}_stats.shp"
    x_train, y_train, points = prepare_classification_data(stats_path=os.path.join(result_folder, shp_name),
                                                           points_path=os.path.join(vector_path, "truth_data.shp"))
    classification_step(x=x_train,
                        y=y_train,
                        month=month,
                        stats_path=os.path.join(result_folder, shp_name),
                        result_folder=result_folder)
    equal_random_stratified(month=month,
                            classified_shp_path=os.path.join(result_folder, f"{month}_classification.shp"),
                            field_name="class_name",
                            number_segments=30,
                            result_folder=result_folder)


if __name__ == "__main__":
    main()
