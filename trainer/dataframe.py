import os
import pandas as pd
from .meter import Meter


class DataFrame():
    def __init__(self):
        self.df = pd.DataFrame(columns=[
            "epoch",
            "training_steps",
            "train:ELBO",
            "train:MSE",
            "train:KLD",
            "test:ELBO",
            "test:MSE",
            "test:KLD",
            "elapsed_time(min)",
        ])
        self.snapshot_filename = "loss.csv"

    def append(self, epoch, meter_train: Meter, meter_test: Meter):
        data = [epoch]
        if meter_train is not None:
            data.append(meter_train.num_updates)
            data.append(meter_train.ELBO.average)
            data.append(meter_train.mean_squared_error.average)
            data.append(meter_train.kl_divergence.average)
        else:
            data.append(0)
            data.append(0)
            data.append(0)
            data.append(0)

        if meter_test is not None:
            data.append(meter_test.ELBO.average)
            data.append(meter_test.mean_squared_error.average)
            data.append(meter_test.kl_divergence.average)
        else:
            data.append(0)
            data.append(0)
            data.append(0)

        data.append(meter_train.elapsed_time)

        series = pd.Series(data, index=self.df.columns)
        self.df = self.df.append(series, ignore_index=True)

    def save(self, root_directory):
        csv_path = os.path.join(root_directory, self.snapshot_filename)
        self.df.to_csv(csv_path, index=False)

    def load(self, root_directory):
        csv_path = os.path.join(root_directory, self.snapshot_filename)
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            return True
        return False