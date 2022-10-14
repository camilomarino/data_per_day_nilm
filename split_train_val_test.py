from pathlib import Path

import typer

from data_per_day import DataPerDay

app = typer.Typer(add_completion=False)

ELECS = [
    "electric water heating appliance",
    "air conditioner",
    "electric vehicle",
    "kettle",
    "fridge",
    "washing machine",
    "microwave",
    "dish washer",
]


@app.command()
def main(
    dataperday_path: Path,
    dataperday_train_path: Path,
    dataperday_val_path: Path,
    dataperday_test_path: Path,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 2,
) -> None:
    """
    It receives a `DataPerDay` and randomly splits it into 3 data sets (train,
    val and test) according to the sizes indicated with `train_size`, `val_size`
    and `test_size`. A seed is set for reproducibility purposes.
    """
    if train_size + val_size + test_size != 1.0:
        raise ValueError

    # load complete dataset
    dataperday_dataset = DataPerDay.load(dataperday_path).get_by(
        elec=ELECS, mantain_others=True
    )

    # split sets
    dataperday_train, dataperday_val_test = dataperday_dataset.train_test_split(
        val_size + test_size, seed=seed
    )
    dataperday_val, dataperday_test = dataperday_val_test.train_test_split(
        test_size / (test_size + val_size), seed=seed
    )

    # save
    dataperday_train.save(dataperday_train_path)
    dataperday_val.save(dataperday_val_path)
    dataperday_test.save(dataperday_test_path)


if __name__ == "__main__":
    app()
