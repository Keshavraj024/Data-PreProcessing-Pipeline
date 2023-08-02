"""
Data Pre-Processing for Machine learning Tasks
"""
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import typing
import logging
import sys


class DataPreprocessing:
    """A class representing Data PreProcessing

    Args:
            data (typing.Union[str, os.PathLike]): file to read the dataset
            strategy (str, optional): Strategy to fill the missing value. Defaults to "mean".
            split_percentage (float, optional): Percentage to split the test dataset. Defaults to 0.2.
            save (bool, optional): Flag to save the date to files. Defaults to True.
    """

    def __init__(
        self,
        data: typing.Union[str, os.PathLike],
        strategy: str = "mean",
        split_percentage: float = 0.2,
        save: bool = False,
    ):
        dataset = pd.read_csv(data)
        self.features = dataset.iloc[:, :-1].values
        self.labels = dataset.iloc[:, -1].values
        self.imputer(strategy=strategy)
        self.onehot_encoding()
        self.label_encoder()
        self.train_test_partition(split_percentage=split_percentage)
        self.feature_scaling()
        if save:
            self.save_dataset()

    def imputer(self, strategy: str = "mean"):
        """Taking care of missing data
        Args:
            strategy (str, optional): Strategy to fill the missing value. Defaults to "mean".
        """
        match strategy:
            case "mean":
                data_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            case "median":
                data_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            case "most_frequent":
                data_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            case "constant":
                y = int(input("Enter an integer for the constant value :"))
                data_imputer = SimpleImputer(
                    missing_values=np.nan, strategy=strategy, fill_value=y
                )

        self.features[:, 1:3] = data_imputer.fit_transform(self.features[:, 1:3])

    def onehot_encoding(self):
        """OneHot encoding of categorical data"""
        Encoder = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
        )
        self.features = Encoder.fit_transform(self.features)

    def label_encoder(self):
        """Encode the labels"""
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

    def train_test_partition(self, split_percentage: float) -> None:
        """Split the data into train and test dataset

        Args:
                split_percentage (float): Percentage to split the test dataset
        """
        if float(split_percentage) < 0 or float(split_percentage) > 1:
            logging.error("Split percentage out of bounds. Defaulting to 0.2")
            split_percentage = 0.2
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=split_percentage, random_state=0
        )

    def feature_scaling(self) -> None:
        """Feature Scale the date by either Normalization or Standardisation"""
        sc = StandardScaler()
        self.X_train[:, 3:5] = sc.fit_transform(self.X_train[:, 3:5])
        self.X_test[:, 3:5] = sc.fit_transform(self.X_test[:, 3:5])

    def save_dataset(self) -> None:
        """Save the csv files to disk"""
        write_to_csv("train_set_x", self.X_train)
        write_to_csv("train_set_y", self.y_train)
        write_to_csv("test_set_x", self.X_test)
        write_to_csv("test_set_y", self.y_test)
        logging.info("Files saved to disk")


def write_to_csv(filename, data) -> None:
    """Write data to a CSV file

    Args:
            filename (str): The name of the CSV file
            data (numpy.ndarray): The data to write to the CSV file
    """
    data = [str(item) if isinstance(item, np.int64) else item for item in data]
    with open(filename + ".csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)


def main():
    """
    Run the data pre-processing pipeline
    """
    if len(sys.argv) > 1:
        DataPreprocessing(
            data=sys.argv[1], strategy=sys.argv[2], split_percentage=sys.argv[3]
        )


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
