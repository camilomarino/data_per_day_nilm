"""
Contains some algorithms to convert NILMTK Datasets to numpy arrays.
"""
import pickle
import sys
from collections import Counter, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
from tqdm import tqdm

from .mapping_labels import LABEL_TO_IDX
from .utils import block_print, enable_print


class PlotAccessorDataPerDay:
    """
    Accessor to plot DataPerDay statics
    """

    def __init__(self, dataset: "DataPerDay"):
        self.dataset = dataset

    def bar_datasets(self, *args, **kwargs):
        """
        Bar plot of number of data of each dataset
        """
        s = pd.Series(self.dataset.differents_datasets()).sort_values(ascending=False)
        ax = s.plot.bar(*args, **kwargs)
        ax.grid(True)
        ax.set_ylabel("Cantidad de muestras")
        ax.set_xlabel("Dataset")
        return ax

    def bar_appliances(self, *args, **kwargs):
        """
        Bar plot of number of data of each dataset
        """
        s = pd.Series(self.dataset.differents_appliances()).sort_values(ascending=False)
        ax = s.plot.bar(*args, **kwargs)
        ax.grid(True)
        ax.set_ylabel("Cantidad de muestras")
        ax.set_xlabel("Appliance")
        return ax

    def bar_elecs(self, most_common: Optional[int] = None, *args, **kwargs):
        """
        Bar plot of number of data of each dataset
        """
        if most_common is not None:
            dataset = self.dataset.dataperday_with_most_common_elecs(most_common)
            len_complete_dataset = len(self.dataset)
            len_most_common_dataset = len(dataset)
            num_others_elecs = len_complete_dataset - len_most_common_dataset
        else:
            dataset = self.dataset

        s = pd.Series(dataset.differents_elecs())
        if most_common is not None:
            s["others"] = num_others_elecs
        s.sort_values(inplace=True, ascending=False)
        ax = s.plot.bar(*args, **kwargs)
        ax.grid(True)
        ax.set_ylabel("Cantidad de muestras")
        ax.set_xlabel("Elecs")
        return ax

    def sample(self, idx: int, ax=None, aggregate: bool = False, *args, **kwargs):
        """
        Plot the day with index idx.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.dataset.X[idx], *args, label="elec", **kwargs)
        if aggregate:
            ax.plot(self.dataset.agg[idx], *args, label="aggregate", **kwargs)
            ax.legend()
        ax.set_title(str(self.dataset.y[idx]))
        return ax


class DataPerDay:
    """
    Class that contains data and labels of nilmtk DataSet separated
    by day.
    This class separate train set and test set using a percentage of
    electric appliances. Also, with this class, is simple to concatenate
    two (or more) DataPerDay only using add operateor (+).

    Attributes
    ----------
        - X: array with day data
        - y: array with the corresponding labels:
                - Appliance name
                - Database name
                - House number
                - Instance relating to the house
                - Date
        - sample_period
        - threshold
    """

    def __init__(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        sample_period: int,
        threshold: int,
        additional_arrays: Optional[Dict[str, np.ndarray]] = None,  # agg, agg_reactive
    ):
        self.X = X
        self.y = y
        self.sample_period = sample_period
        self.threshold = threshold
        self.plot = PlotAccessorDataPerDay(self)
        self.additional_arrays = (
            additional_arrays if additional_arrays is not None else {}
        )
        self._check_sizes()

    def _check_sizes(self):
        """
        Check that every size is the same for all the arrays.
        """
        sizes = [len(self.X), len(self.y)]
        for array in self.additional_arrays.values():
            sizes.append(len(array))
        if len(set(sizes)) != 1:
            raise ValueError("Sizes of arrays are not the same")

    def __repr__(self):
        string = (
            f"""(<<DataPerDay>>: \n\t{self.X.shape[0]} days, """
            f"""\n\t{len(self.differents_datasets())} differents datasets, """
            f"""\n\t{len(self.differents_appliances())} differents appliances, """
            f"""\n\t{len(np.unique(self.y[:, 0]))} differents elecs type, """
            f"""\n\t{self.sample_period} sample period, """
            f"""\n\tX: {self.X}, \n\ty: {self.y}\n)\n"""
        )
        return string

    def __add__(self, other: Union["DataPerDay", None]) -> "DataPerDay":
        """
        Concatenate DataPerDay. Returns a new instance of
        DataPerDay with the samples of both of them.
        """
        if other is None:
            return self
        assert self.sample_period == other.sample_period
        assert self.threshold == other.threshold
        X = np.concatenate((self.X, other.X), axis=0)
        y = np.concatenate((self.y, other.y), axis=0)
        additional_arrays = {}
        for key in self.additional_arrays:
            if key in other.additional_arrays:
                # if doesn't exist in other, it doesn't exist in the result
                additional_arrays[key] = np.concatenate(
                    (self.additional_arrays[key], other.additional_arrays[key]), axis=0
                )
        sample_period = self.sample_period
        threshold = self.threshold
        result = type(self)(
            X=X,
            y=y,
            sample_period=sample_period,
            threshold=threshold,
            additional_arrays=additional_arrays,
        )
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __getitem__(self, idx) -> "DataPerDay":
        """
        Returns a subset of index idx. The indexing is
        equal to numpy.
        """
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        X = self.X[idx]
        y = self.y[idx]
        sample_period = self.sample_period
        threshold = self.threshold
        additional_arrays = {}
        for key in self.additional_arrays:
            additional_arrays[key] = self.additional_arrays[key][idx]
        result = type(self)(
            X=X,
            y=y,
            sample_period=sample_period,
            threshold=threshold,
            additional_arrays=additional_arrays,
        )
        return result

    def __len__(self):
        """
        Return number of data samples
        """
        return len(self.X)

    def random_choice(self, size: int, seed: int = 1) -> "DataPerDay":
        """
        Returns a DataPerDay that contains a subset of size=size
        """
        np.random.seed(seed)
        idx = np.random.choice(len(self), size=size, replace=False)
        return self[idx]

    def shuffle(self, seed: int = 1) -> "DataPerDay":
        """
        Returns a DataPerDay view with new order of samples.
        """
        new_idx = np.random.permutation(len(self))
        return self[new_idx]

    @property
    def data_elec(self) -> np.ndarray:
        """ "
        Para propositos de usar en desagregacion. Utilizar el nombre X
        resulta confuso en desagregacion para los datos desagregados.
        """
        return self.X

    @property
    def data_agg(self) -> np.ndarray:
        return self.additional_arrays["agg"]

    @property
    def data_agg_reactive(self) -> np.ndarray:
        return self.additional_arrays["agg_reactive"]

    @property
    def agg(self) -> np.ndarray:
        return self.data_agg

    @property
    def agg_reactive(self) -> np.ndarray:
        return self.data_agg_reactive

    def binary_label(
        self, type_of_appliance: str = "electric water heating appliance"
    ) -> np.ndarray:
        """
        Returns and calculates the binary label, where the
        electric water heating appliance corresponds to 1.
        """
        return self.y[:, 0] == type_of_appliance

    def labels(
        self,
        like_string: bool = False,
    ) -> np.ndarray:
        """
        Return the labels of the dataset.

        Parameters
        ----------
        like_string (bool): if True return string name of elec
                            if False return an index label
        """
        if like_string:
            return self.y[:, 0]
        else:
            elecs = self.y[:, 0]

            label_to_idx_vectorized = np.vectorize(LABEL_TO_IDX.get)
            labels = label_to_idx_vectorized(elecs)
            return labels

    def weights(self, normalized: bool = True) -> np.ndarray:
        """
        Returns the inverse of probability of each elec in one hot mode.

        Parameters
        ----------
            normalized (bool): if is true normalize weight to sum 1
        """
        # create list of tuple with (idx_elec, #of_this_elec_in_dataset)
        list_counts = [
            (LABEL_TO_IDX[k], v) for (k, v) in self.differents_elecs().items()
        ]

        # separate idx and counts
        idx, counts = np.array(list_counts).T

        # calculate probability of each elec in the dataset
        density = np.zeros(len(LABEL_TO_IDX), dtype=np.float32)
        density[idx] = counts / counts.sum()

        # calculate weigths to apply in each class
        weight = density.copy()
        weight[idx] = 1 / weight[idx]
        if normalized:
            weight /= weight.sum()
        return weight

    def groups(self) -> np.ndarray:
        """
        Returns an array of the same size as the amount of data, with an appliance identifier.
        Especially useful to use as `group` parameter in cross validation in sklearn.
        """
        hash_array = np.vectorize(hash)
        result = hash_array(self.y[:, 0:-1])
        _, groups = np.unique(result, axis=0, return_inverse=True)
        return groups

    def differents_elecs(self) -> Counter:
        """
        Returns a Counter of differents elecs.
        """
        return Counter(self.labels(like_string=True))

    def differents_appliances(self) -> Counter:
        """
        Returns a Counter of differents appliacnces.
        """
        return Counter(map(tuple, tuple(self.y[:, :-1])))

    def differents_datasets(self) -> Counter:
        """
        Returns a Counter of diferente appliances.
        """
        return Counter(self.y[:, 1])

    def dataperday_with_min_appliances_per_elec(self, min_appliance: int = 2):
        serie = pd.Series(
            Counter([elec[0] for elec in self.differents_appliances().keys()])
        )
        elecs = serie.index[serie > min_appliance]
        return self.get_elec(elecs)

    def dataperday_with_most_common_elecs(self, num_most_common: int) -> "DataPerDay":
        """
        Get the idx of most common elecs and return a DataPerDay with
        only this elecs.
        """
        elecs = self.differents_elecs().most_common(num_most_common)
        idx = np.isin(y[:, 0], elecs)
        return self[idx]

    def summary_elec_dataset(
        self,
        num_most_common: Optional[int] = None,
        add_sum_column: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a data frame table containing the number of occurrences of
        each elec in each dataset.

        Parameters
        ----------
            num_most_common (int): return only the most_common elecs
        """

        if num_most_common is None:
            dataset = self
        else:
            dataset = self.dataperday_with_most_common_elecs(num_most_common)

        df = pd.DataFrame(dataset.y[:, 0:2], columns=["elec", "dataset"])
        df = df.pivot_table(index="elec", columns="dataset", aggfunc=len, fill_value=0)

        if add_sum_column:
            df["sum"] = df.sum(axis=1)

        return df

    def train_test_split(
        self, test_size: float = 0.25, seed: int = 0
    ) -> Tuple["DataPerDay", "DataPerDay"]:
        """
        Returns a namedtuple that contains two DataPerDay. First
        corresponds with the train set and the second corresponds with
        the test set.

        The split is about appliances. Select test_size appliances to test
        and 1-test_size to train.
        """
        sample_period = self.sample_period
        threshold = self.threshold

        np.random.seed(seed)

        Xs_train = []
        ys_train = []
        additional_arrays_train = {}
        for key in self.additional_arrays.keys():
            additional_arrays_train[key] = []

        Xs_test = []
        ys_test = []
        additional_arrays_test = {}
        for key in self.additional_arrays.keys():
            additional_arrays_test[key] = []

        for elec_name in tqdm(self.differents_elecs().keys()):
            dataperday_elec = self.get_elec(elec_name)
            X_elec = dataperday_elec.X
            y_elec = dataperday_elec.y
            additional_arrays_elec = {}
            for key in self.additional_arrays.keys():
                additional_arrays_elec[key] = dataperday_elec.additional_arrays[key]
            if len(dataperday_elec.differents_appliances()) <= 1:
                Xs_train.append(X_elec)
                ys_train.append(y_elec)
                for key in self.additional_arrays.keys():
                    additional_arrays_train[key].append(additional_arrays_elec[key])
            else:
                groups_elec = dataperday_elec.groups()
                gss = GroupShuffleSplit(n_splits=1, train_size=1 - test_size)
                idx_train, idx_test = next(gss.split(X_elec, groups=groups_elec))
                Xs_train.append(X_elec[idx_train])
                ys_train.append(y_elec[idx_train])
                for key in self.additional_arrays.keys():
                    additional_arrays_train[key].append(
                        additional_arrays_elec[key][idx_train]
                    )

                Xs_test.append(X_elec[idx_test])
                ys_test.append(y_elec[idx_test])
                for key in self.additional_arrays.keys():
                    additional_arrays_test[key].append(
                        additional_arrays_elec[key][idx_test]
                    )

        Xs_train = np.concatenate(Xs_train)
        ys_train = np.concatenate(ys_train)
        for key in self.additional_arrays.keys():
            additional_arrays_train[key] = np.concatenate(additional_arrays_train[key])

        Xs_test = np.concatenate(Xs_test)
        ys_test = np.concatenate(ys_test)
        for key in self.additional_arrays.keys():
            additional_arrays_test[key] = np.concatenate(additional_arrays_test[key])

        dataperday_train = DataPerDay(
            X=Xs_train,
            y=ys_train,
            sample_period=sample_period,
            threshold=threshold,
            additional_arrays=additional_arrays_train,
        )
        dataperday_test = DataPerDay(
            X=Xs_test,
            y=ys_test,
            sample_period=sample_period,
            threshold=threshold,
            additional_arrays=additional_arrays_test,
        )
        TrainTestDataPerDay = namedtuple("TrainTestDataPerDay", "train test")

        return TrainTestDataPerDay(dataperday_train, dataperday_test)

    def resample(
        self, new_sample_periods: Union[int, List[int]]
    ) -> "DataPerDayCollection":
        """
        new_sample_periods: in seconds
        """
        result = DataPerDayCollection()
        if isinstance(new_sample_periods, int):
            new_sample_periods = [new_sample_periods]
        for new_sample_period in new_sample_periods:
            assert new_sample_period % self.sample_period == 0
            X = self.X
            y = self.y
            additional_arrays = {}
            for key in self.additional_arrays.keys():
                additional_arrays[key] = self.additional_arrays[key]
            threshold = self.threshold
            factor = new_sample_period // self.sample_period
            shape = X.shape
            X = X.reshape((shape[0], shape[1] // factor, -1)).mean(axis=2)
            for key in self.additional_arrays.keys():
                additional_arrays[key] = (
                    additional_arrays[key]
                    .reshape((shape[0], shape[1] // factor, -1))
                    .mean(axis=2)
                )
            new_array = type(self)(
                X=X,
                y=y,
                sample_period=new_sample_period,
                threshold=threshold,
                additional_arrays=additional_arrays,
            )
            result[new_sample_period] = new_array
        return result

    def get_n_per_elec(self, num_samples: Optional[int] = None, seed: int = 0):
        """
        Toma `num_samples` de cada electrodomestico y devuelve un DataPerDay.
        Util para entrenar un modelo balanceado con las mismas muestras de todos
        los electrodomesticos.

        Si `num_samples` no es dado se usa la cantidad de muestras del electrodomestico
        que tiene mas.

        Se busca que haya la menor cantidad de muestras repetidas posibles.
        """
        if num_samples is None:
            # number of samples of the most common electhe elec
            num_samples = max(self.differents_elecs().values())

        sample_period = self.sample_period
        threshold = self.threshold

        np.random.seed(seed)

        Xs = []
        ys = []
        additional_arrays = {}
        for key in self.additional_arrays.keys():
            additional_arrays[key] = []

        for elec_name in tqdm(self.differents_elecs().keys()):
            dataperday_elec = self.get_elec(elec_name)
            X = dataperday_elec.X
            y = dataperday_elec.y
            additional_arrays_elec = {}
            for key in self.additional_arrays.keys():
                additional_arrays_elec[key] = dataperday_elec.additional_arrays[key]

            # Seleccion `num_samples` sin remplazo para evitar repetidos
            idx_1 = np.random.choice(
                np.arange(X.shape[0]), min(num_samples, len(X)), replace=False
            )
            Xs.append(X[idx_1])
            ys.append(y[idx_1])
            for key in self.additional_arrays.keys():
                additional_arrays[key].append(additional_arrays_elec[key][idx_1])

            idx_2 = np.random.randint(0, X.shape[0], num_samples - len(idx_1))
            Xs.append(X[idx_2])
            ys.append(y[idx_2])
            for key in self.additional_arrays.keys():
                additional_arrays[key].append(additional_arrays_elec[key][idx_2])

        Xs = np.concatenate(Xs)
        ys = np.concatenate(ys)
        for key in self.additional_arrays.keys():
            additional_arrays[key] = np.concatenate(additional_arrays[key])

        return DataPerDay(
            X=Xs,
            y=ys,
            sample_period=sample_period,
            threshold=threshold,
            additional_arrays=additional_arrays,
        )

    def get_by(
        self,
        elec=None,
        dataset=None,
        house=None,
        instance=None,
        date=None,
        mantain_others: bool = False,
    ):
        """
        Selects item by elec, dataset, house, instance or date. More than one
        attribute is also allowed.
        """
        idx = np.ones(len(self), dtype=bool)
        if elec is not None:
            idx &= np.isin(self.y[:, 0], elec)
        if dataset is not None:
            idx &= np.isin(self.y[:, 1], dataset)
        if house is not None:
            idx &= np.isin(self.y[:, 2], house)
        if instance is not None:
            idx &= np.isin(self.y[:, 3], instance)
        if date is not None:
            idx &= np.isin(self.y[:, 4], date)

        dataset = self
        dataset.y = self.y.copy()

        elecs_dataset = dataset[idx]

        # los que no no cumplen la condicion se los etiqueta como other
        if mantain_others:
            others_dataset = dataset[~idx]
            others_dataset.y[:, 0] = "unknown"

            result = elecs_dataset + others_dataset
        else:
            result = elecs_dataset

        return result

    def drop_by(self, elec=None, dataset=None, house=None, instance=None, date=None):
        """
        Delete item by elec, dataset, house, instance or date. More than one
        attribute is also allowed.
        """

        idx = np.ones(len(self), dtype=bool)
        if elec is not None:
            idx &= np.isin(self.y[:, 0], elec)
        if dataset is not None:
            idx &= np.isin(self.y[:, 1], dataset)
        if house is not None:
            idx &= np.isin(self.y[:, 2], house)
        if instance is not None:
            idx &= np.isin(self.y[:, 3], instance)
        if date is not None:
            idx &= np.isin(self.y[:, 4], date)

        return self[~idx]

    def get_elec(self, elec: Union[str, List[str]]) -> "DataPerDay":
        return self.get_by(elec=elec)

    def drop_elec(self, elec: Union[str, List[str]]) -> "DataPerDay":
        return self.drop_by(elec=elec)

    def drop_unknown(self):
        # import ipdb; ipdb.set_trace()
        return self.drop_elec("unknown")

    def drop_aggregate_nan(self):
        """ "
        Elimina los datos que tienen nan en su agregado.

        La idea es usar estos datos para clasificar pero no para desagregar.
        """
        idx = (
            np.isnan(self.agg).sum(axis=1) == 0
        )  # indices que tienen no tienen ningun nan
        return self[idx]

    def clean_signals(self):
        """ "
        Elimina los datos que cumplen:
            - potencia agregada tenga un pico que sea mayor a 15kW
            - potencia desagregada tenga un pico que sea mayor a 10kW
            - tenga más de 1 muestras con valores mayores en la desagregada
                que en la agregada
            - los datos con potencia menor a -50 Watts (los que estan entre -50 y 0
                los mando a 0)
        """
        idx = np.ones(len(self), dtype=bool)
        idx &= self.agg.max(axis=1) < 15_000  # agg < 15kW
        idx &= self.X.max(axis=1) < 10_000  # elec < 10kW
        idx &= (
            (self.X > self.agg).sum(axis=1)
        ) < 2  # agg > elec en casi todas las muestras
        idx &= ~((self.X < -15).sum(axis=1) > 0)  # potencias negativas
        if "agg" in self.additional_arrays.keys():
            idx &= ~(
                (self.additional_arrays["agg"] < -15).sum(axis=1) > 0
            )  # potencias negativas

        result = self[idx]
        result.X = np.clip(
            result.X, 0, None
        )  # mando a 0 las potencias negativas pequenas
        if "agg" in self.additional_arrays.keys():
            result.additional_arrays["agg"] = np.clip(
                result.additional_arrays["agg"], 0, None
            )  # mando a 0 las potencias negativas pequenas
        return result

    def change_elec_name(self, name_map: Dict[str, str]) -> "DataPerDay":
        """
        Change names of the elecs usign the mapping in name_map
        """
        elecs_names = self.y[:, 0]
        for old_name, new_name in name_map.items():
            elecs_names[elecs_names == old_name] = new_name
        return self

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save the object to memory.
        """
        with open(path, "wb") as fp:
            pickle.dump(
                {
                    "X": self.X,
                    "y": self.y,
                    "sample_period": self.sample_period,
                    "threshold": self.threshold,
                    "additional_arrays": self.additional_arrays,
                },
                fp,
            )

    @classmethod
    def load(clf, path: str) -> "DataPerDay":
        """
        Load object from memory and returns it.
        """
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        return clf(
            X=data["X"],
            y=data["y"],
            sample_period=data["sample_period"],
            threshold=data["threshold"],
            additional_arrays=data["additional_arrays"],
        )

    @classmethod
    def concatenate(clf, datas_per_day: list) -> "DataPerDay":
        """
        Recives a list of DataPerDay and concatenates its in a unique
        DataPerDay.
        """
        result = None
        for data in datas_per_day:
            result += data
        return result


class DataPerDayCollection(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_test_split(
        self, *args, **kwarg
    ) -> Tuple["DataPerDayCollection", "DataPerDayCollection"]:
        train_collection = DataPerDayCollection(
            map(
                lambda kv: (kv[0], kv[1].train_test_split(*args, **kwarg).train),
                self.items(),
            )
        )
        test_collection = DataPerDayCollection(
            map(
                lambda kv: (kv[0], kv[1].train_test_split(*args, **kwarg).test),
                self.items(),
            )
        )
        return train_collection, test_collection

    def train_split(self, *args, **kwarg):
        return DataPerDayCollection(
            map(
                lambda kv: (kv[0], kv[1].train_test_split(*args, **kwarg).train),
                self.items(),
            )
        )

    def test_split(self, *args, **kwarg):
        return DataPerDayCollection(
            map(
                lambda kv: (kv[0], kv[1].train_test_split(*args, **kwarg).test),
                self.items(),
            )
        )


def _separate_df_per_day(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Separate a data frame into a list of data frames.
    Each item in the list is a data frame with items
    the same day.
    Extracted from: https://stackoverflow.com/a/21605687
    """
    DFList = []
    for group in df.groupby(df.index.date):
        DFList.append(group[1])
    return DFList


def convert_nilmtkh5_to_dataperday(
    nilmtk_dataset,
    threshold: int = 50,
    sample_period: int = 60,  # in seconds, multiple of seconds in a day
    verbose: bool = True,
) -> DataPerDay:
    """
    Divide the electrical consumption series of appliances
    (site meters not used) by days and generates training
    vectors and labels for a sample period.
    Days whose maximum is below the threshold are discarded.

    Returns a DataPerDay with:
        - X: array with training data
        - y: array with the corresponding labels:
                - Appliance name
                - Database name
                - House number
                - Instance relating to the house
                - Date
        - sample_period
        - threshold
    """
    dataset_name = nilmtk_dataset.metadata["name"]
    X = []
    agg = []
    agg_reactive = []
    y = []
    for building in tqdm(
        nilmtk_dataset.buildings.keys(),
        disable=not verbose,
        unit="house",
        desc=f"{dataset_name:8}",
        file=sys.stdout,
    ):
        site_meter = nilmtk_dataset.buildings[building].elec.mains()
        if site_meter is None:
            print(f"building {building} of {dataset_name} doesn't have site meter.")
            df_agg_all = pd.DataFrame()  # dummy DataFrame
        else:
            df_agg_all = next(
                site_meter.load(
                    sample_period=sample_period, resample=True, verbose=False
                )
            )
            # add suffix
            # https://stackoverflow.com/a/57740435
            df_agg_all = (
                df_agg_all.stack(level=1)
                .add_suffix("_agg")
                .unstack()
                .dropna(how="all", axis=1)
            )

        for elec in nilmtk_dataset.buildings[building].elec.all_meters():
            if not elec.is_site_meter():

                if len(elec.appliances) != 1:
                    # if is multi appliances or not labeled
                    # is omitted
                    continue

                # load data
                block_print()  # nilmtk has some unnecessary footprints
                df_elec_all = next(
                    elec.load(sample_period=sample_period, resample=True, verbose=False)
                )
                enable_print()
                if len(df_elec_all) == 0:
                    continue

                elec_name = elec.appliances[0].type["type"]
                elec_instance = elec.instance()

                # add suffix
                # https://stackoverflow.com/a/57740435
                df_elec_all = (
                    df_elec_all.stack(level=1)
                    .add_suffix("_elec")
                    .unstack()
                    .dropna(how="all", axis=1)
                )

                df_all = df_elec_all.join(df_agg_all, how="left")

                # verifico que tenga las mediciones correctas
                if (("power_elec", "active") not in df_all) and (
                    ("power_elec", "apparent") not in df_all
                ):
                    continue

                # verifico que tenga las mediciones correctas
                if (("power_agg", "active") not in df_all) and (
                    ("power_agg", "apparent") not in df_all
                ):
                    continue

                for df in _separate_df_per_day(df_all):
                    seconds_in_a_day = 60 * 60 * 24
                    if (
                        len(df) == seconds_in_a_day / sample_period
                        and df["power_elec"].values.max() > threshold
                    ):
                        if (
                            "power_elec",
                            "active",
                        ) in df_all:  # active power is the priority
                            X.append(df["power_elec", "active"].to_numpy())
                        elif ("power_elec", "apparent") in df_all:
                            X.append(df["power_elec", "apparent"].to_numpy())

                        if (
                            "power_agg",
                            "active",
                        ) in df_all:  # active power is the priority
                            agg.append(df["power_agg", "active"].to_numpy())
                        elif ("power_agg", "apparent") in df_all:
                            agg.append(df["power_agg", "apparent"].to_numpy())
                        else:
                            agg.append(np.full(len(X[0]), np.nan))

                        if ("power_agg", "reactive") in df_all:
                            agg_reactive.append(df["power_agg", "reactive"].to_numpy())
                        else:
                            agg_reactive.append(np.empty_like(agg[-1]))

                        y.append(
                            (
                                elec_name,
                                dataset_name,
                                building,
                                elec_instance,
                                df.index[0].date(),
                            )
                        )
                        import ipdb

                        ipdb.set_trace()
    X, agg, agg_reactive, y = (
        np.array(X, dtype=np.float32),
        np.array(agg, dtype=np.float32),
        np.array(agg_reactive, dtype=np.float32),
        np.array(y, dtype=np.object),
    )
    if len(agg_reactive) == len(agg):
        additional_arrays = {"agg": agg, "agg_reactive": agg_reactive}
    else:
        additional_arrays = {"agg": agg}
        print("agg_reactive is not available or has different length than agg")

    return (
        DataPerDay(
            X=X,
            y=y,
            sample_period=sample_period,
            threshold=threshold,
            additional_arrays=additional_arrays,
        )
        .drop_unknown()
        .drop_by(elec="sockets")
    )
