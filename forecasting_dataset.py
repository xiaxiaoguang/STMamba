import os
import torch
from torch.utils.data import Dataset
import numpy as np

class ForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, processed_data, index ,mode: str) -> None:
        """Init the dataset in the forecasting stage.

        Args:
            data_file_path (str): data file path.
            index_file_path (str): index file path.
            mode (str): train, valid, or test.
            seq_len (int): the length of long term historical data.
        """

        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"

        self.data = torch.from_numpy(processed_data).float()# read index
        self.index = torch.from_numpy(index[mode])

        # for idx,a in enumerate(self.index):
        #     if a[0]>=seq_len:
        #         self.index = self.index[idx:]
        #         break

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str,label_file_path):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        if label_file_path is not None and not os.path.isfile(label_file_path):
            raise FileNotFoundError("BasicTS can not find label file {0}".format(label_file_path))
        
    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])
        history_data = self.data[idx[0]:idx[1]]     
        future_data = self.data[idx[1]:idx[2]]        

        return history_data,future_data

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)
