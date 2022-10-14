import os
from pathlib import Path

import typer
from nilmtk import DataSet

from data_per_day import DataPerDay, convert_nilmtkh5_to_dataperday

app = typer.Typer(add_completion=False)


@app.command()
def main(datasets_h5_path: Path, datasets_per_day_path: Path):
    """
    Given a folder `datasets_h5_path` containing datasets in nilmtk format (h5)
    converts them into DataPerDay format and saves them in another folder
    `datasets_per_day_path`.

    In addition, it creates a file in DataPerDay format containing the data of
    ALL datasets. The name of this file is `merged_dataset.pickle`.

    To convert the data, it takes the series of all appliances from all datasets
    in nilmtk format and keeps only the days with a maximum power greater than
    50 watts. It also resamples the data every 60 seconds and discards days with
    missing data.
    """

    nilmtk_datasets = [
        Path(datasets_h5_path, file)
        for file in os.listdir(datasets_h5_path)
        if file.endswith(".h5")
    ]
    print(f"The datasets are: {nilmtk_datasets}")
    print("-" * 70)
    for path_h5 in nilmtk_datasets:
        dataset = DataSet(path_h5)
        name = dataset.metadata["name"]
        path_pickle = Path(datasets_per_day_path, f"data_per_day_{name}.pickle")
        if path_pickle.is_file():
            print(f"Using {name} from cache in {path_pickle}")
        else:
            data = convert_nilmtkh5_to_dataperday(
                dataset, threshold=50, sample_period=60
            )
            data.save(path=path_pickle)

    print("-" * 70)

    merged_dataset_path = Path(datasets_per_day_path, "merged_dataset.pickle")
    print(f"Merging all the datasets and saving to {merged_dataset_path}")
    dataperday_datasets = [
        Path(datasets_per_day_path, file)
        for file in os.listdir(datasets_per_day_path)
        if file.endswith(".pickle") and "merged" not in file
    ]
    dataset = None
    for dataperday_dataset in dataperday_datasets:
        print(f"Loading {dataperday_dataset}")
        dataset += DataPerDay.load(dataperday_dataset)

    # borro para estar del lado seguro, no deberia haber
    dataset = dataset.drop_unknown().drop_by(elec="sockets")
    dataset.save(merged_dataset_path)
    print("-" * 70)


if __name__ == "__main__":
    app()
