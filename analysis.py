import geopandas as gpd
import configparser
import os


def analysis(season, all_dataset, lake, both_lake, error_field):
    print(f"In {season} image were classified {str(len(lake) + len(both_lake))} lakes")
    classification_error = []
    segmentation_error = []
    for index, row in all_dataset.iterrows():
        if row[error_field] == "Cl" and row["Legend"] != "N":
            classification_error.append("")
        elif row[error_field] == "Se" and row["Legend"] != "N":
            segmentation_error.append("")
    print(f"In {season} image {str(len(classification_error))} lakes have classification error")
    print(f"In {season} image {str(len(segmentation_error))} lakes have segmentation error")


def execute(shp):
    shp_df = gpd.read_file(shp)
    none = []
    winter = []
    summer = []
    both = []
    for index, row in shp_df.iterrows():
        if row["Legend"] == "W":
            winter.append("")
        elif row["Legend"] == "S":
            summer.append("")
        elif row["Legend"] == "B":
            both.append("")
        elif row["Legend"] == "N":
            none.append("")
    print(f"{str(len(none))} lakes not classified")
    print(f"{str(len(summer))} lakes classified only in summer")
    print(f"{str(len(winter))} lakes classified only in winter")
    print(f"{str(len(both))} lakes classified both in summer and winter")
    analysis(season="summer",
             all_dataset=shp_df,
             lake=summer,
             both_lake=both,
             error_field="Serror")
    analysis(season="winter",
             all_dataset=shp_df,
             lake=winter,
             both_lake=both,
             error_field="Werror")
    print(f"{str(len(summer) + len(winter) + len(both))} lakes were classified combining both images")


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")
    vector_path = config.get("folders", "vector")
    lake_points = os.path.join(vector_path, "Lake_points.shp")
    execute(lake_points)


if __name__ == "__main__":
    main()
