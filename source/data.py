
import os, sys
from imports import *

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        config, 
        df_path, data_path
        , augment = False
    ):
        self.config = config
        self.df_path, self.data_path = df_path, data_path
        self.augment = augment
        self.__len__()

    def __len__(self):
        self.df = pandas.read_csv(self.df_path)
        return len(self.df)

    def drop_lead(self, ecg):
        if random.random() >= 0.5:
            ecg[np.random.randint(len(self.config.ecg_leads)), :] = 0.
        return ecg

    def __getitem__(self, index):
        row = self.df.iloc[index]

        ecg, demographic = np.load("{}/{}.npy".format(self.data_path, row["id"]))[self.config.ecg_leads, :], encode_age(row["age"], augment = False) + encode_sex(row["sex"], augment = False)
        ecg = fix_ecg_length(ecg, self.config.ecg_length)
        if self.augment:
            ecg = self.drop_lead(ecg)
        ecg, demographic = torch.tensor(ecg).float(), torch.tensor(demographic).float()

        if not self.config.is_multilabel:
            label = row["label"]
            return (ecg, demographic), row["r_count"], label
        else:
            label = row[[column for column in list(row.index) if "label_" in column]].values.astype("float64")
            return (ecg, demographic), row["r_count"], label