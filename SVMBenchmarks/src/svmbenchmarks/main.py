import logging
import lzma
import os
import tempfile
import threading
from typing import Optional, Callable, Tuple

import math
import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.protoclasses.grcp_service import GRCPService
from numpy.random import RandomState
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

directory_file_descriptor = tempfile.TemporaryDirectory()
directory_name = directory_file_descriptor.name
lock = threading.Lock()


def download_slice_localization_data():
    """
    Downloads the slice localization data from a specified URL and saves it locally.

    :return: None
    """
    if not os.path.exists(os.path.join(directory_name, "slice_localization_data.csv")):
        print(f"{os.path.join(directory_name, 'slice_localization_data.csv')} not found. Downloading...")
        url = "http://mopta-executables.s3-website.eu-north-1.amazonaws.com/slice_localization_data.csv.xz"
        print(f"Downloading {url}")
        import requests
        response = requests.get(url, verify=False)

        # save the .xz file
        with open(os.path.join(directory_name, "slice_localization_data.csv.xz"), "wb") as out:
            out.write(response.content)
        # unpack the data
        with lzma.open(os.path.join(directory_name, "slice_localization_data.csv.xz"), "rb") as f, open(
                os.path.join(directory_name, "slice_localization_data.csv"), "wt"
        ) as out:
            out.write(f.read().decode("utf-8"))

        print("Downloaded slice_localization_data.csv")
    if not os.path.exists(os.path.join(directory_name, "CT_slice_X.npy")):
        data = np.genfromtxt(
            os.path.join(directory_name, "slice_localization_data.csv"),
            delimiter=",",
            skip_header=1,
        )
        X = data[:, :385]
        y = data[:, -1]
        np.save(os.path.join(directory_name, "CT_slice_X.npy"), X)
        np.save(os.path.join(directory_name, "CT_slice_y.npy"), y)
    X = np.load(os.path.join(directory_name, "CT_slice_X.npy"))
    y = np.load(os.path.join(directory_name, "CT_slice_y.npy"))
    # return copies of the data
    return X.copy(), y.copy()


def load_data_388():
    """
    _load_data()
    -----------

    This method is used to load data for CT slice localization. It downloads the data if necessary and processes it for further use.

    :return: A tuple containing the features (X) and labels (y) of the CT data. X is a numpy array of shape (n_samples, n_features) and y is a numpy array of shape (n_samples,). The features
    * and labels are scaled using MinMaxScaler.

    Example usage:

        >>> X, y = load_data_388()
    """

    X, y = download_slice_localization_data()
    X = MinMaxScaler().fit_transform(X)
    y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
    idxs = RandomState(388).choice(np.arange(len(X)), min(500, len(X)), replace=False)
    half = len(idxs) // 2
    x_train = X[idxs[:half]]
    x_test = X[idxs[half:]]
    y_train = y[idxs[:half]]
    y_test = y[idxs[half:]]

    return x_train, y_train, x_test, y_test


def load_data_53(
        n_features: Optional[int] = None,
):
    X, y = download_slice_localization_data()
    X -= X.min(axis=0)
    X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
    X = X / (X.max(axis=0) - X.min(axis=0))
    X = 2 * X - 1
    assert X.min() == -1 and X.max() == 1

    # Standardize targets
    y = (y - y.mean()) / y.std()

    # Only keep 10,000 data points and n_features features
    shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[:10_000]  # Use seed 0
    X, y = X[shuffled_indices], y[shuffled_indices]

    if n_features is not None:
        # Use Xgboost to figure out feature importances and keep only the most important features
        xgb = XGBRegressor(max_depth=8).fit(X, y)
        inds = (-xgb.feature_importances_).argsort()
        X = X[:, inds[:n_features]]

    # Train/Test split on a subset of the data
    train_n = int(math.floor(0.50 * X.shape[0]))
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:], y[train_n:]

    return train_x, train_y, test_x, test_y


class SvmServiceServicer(GRCPService):
    """
    This class is a GRCP service for SVM evaluation.

    Attributes:
        - X (numpy.ndarray): Input data for training.
        - y (numpy.ndarray): Target values for training.
        - _X_train (numpy.ndarray): Input data for training, subset of X.
        - _X_test (numpy.ndarray): Input data for testing, subset of X.
        - _y_train (numpy.ndarray): Target values for training, subset of y.
        - _y_test (numpy.ndarray): Target values for testing, subset of y.

    Methods:
        - __init__(self): Initializes the SVM service.
        - evaluate_point(self, request: BenchmarkRequest, context) -> EvaluationResult: Evaluates a point using SVM.

    """

    def __init__(
            self
    ):
        super().__init__(port=50058)
        self.data_initialized = None

    def initialize_data(
            self,
            data_loader: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = load_data_388
    ):
        self._X_train, self._y_train, self._X_test, self._y_test = data_loader()

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        """
        evaluate_point Method

        :param request: An instance of the BenchmarkRequest class, representing the request to evaluate a point.
        :param context: The context in which the evaluation is being performed.
        :return: An instance of the EvaluationResult class, representing the result of the evaluation.

        This method evaluates the given point using an SVM regression model. It first extracts the values from the request's point, and applies transformations to them to calculate the SVM model
        * parameters. It then fits the SVM regression model using the transformed training data and evaluates it against the transformed test data. The evaluation result, represented as the
        * root mean square error (RMSE), is returned as an instance of the EvaluationResult class.

        Please note that this method assumes that the benchmark name in the request is "svm". If the benchmark name is different, an assertion error will occur.
        """
        valid_benchmark_names = ['svm', 'svmmixed']
        assert request.benchmark.name in valid_benchmark_names, f"Invalid benchmark name: {request.benchmark.name}. Expected one of {valid_benchmark_names}"
        if request.benchmark.name == 'svmmixed':
            loader = load_data_53
        elif request.benchmark.name == 'svm':
            loader = load_data_388
        else:
            raise ValueError(
                f"Invalid benchmark name: {request.benchmark.name}. Expected one of {valid_benchmark_names}"
            )

        with lock:
            if self.data_initialized is None or self.data_initialized != loader:
                self.initialize_data(loader)
                self.data_initialized = loader

        x = [v.value for v in request.point.values]
        x = np.array(x).squeeze()
        C = 0.01 * (500 ** x[-1])
        gamma = 0.1 * (30 ** x[-2])
        epsilon = 0.01 * (100 ** x[-3])
        if loader == load_data_53:
            inds_selected = np.where(x[np.arange(len(x) - 3)] == 1)[0]
            if len(inds_selected) == 0:
                return EvaluationResult(
                    value=1.0
                )
            else:
                _x_fit = self._X_train[:, inds_selected]
                _x_pred = self._X_test[:, inds_selected]
        elif loader == load_data_388:
            length_scales = np.exp(4 * x[:-3] - 2)
            _x_fit = self._X_train / length_scales
            _x_pred = self._X_test / length_scales

        svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
        svr.fit(_x_fit, self._y_train)
        pred = svr.predict(_x_pred)
        error = np.sqrt(np.mean(np.square(pred - self._y_test)))
        result = EvaluationResult(
            value=float(error)
        )
        return result


def serve():
    logging.basicConfig()
    svm = SvmServiceServicer()
    svm.serve()


if __name__ == '__main__':
    serve()
