
import os, sys
from imports import *

class ECGDatasetSeg(torch.utils.data.Dataset):
    def __init__(self, 
        config, 
        df_path, data_path
    ):
        self.config = config
        self.df_path, self.data_path = df_path, data_path
        self.__len__()

    def __len__(self):
        self.ori_df = pandas.read_csv(self.df_path)
        self.df = pandas.concat([self.ori_df]*12).reset_index()
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        ecg, mask = np.load("{}/{}.npy".format(self.data_path, row["id"]))[[index // len(self.ori_df)], :], np.transpose(np.load("{}_mask/mask_{}.npy".format(self.data_path, row["id"]))[index // len(self.ori_df), :])
        ecg, mask = fix_length(ecg, self.config.ecg_length), fix_length(mask, self.config.ecg_length)
        ecg, mask = torch.tensor(ecg).float(), torch.tensor(mask).float()

        return ecg, mask