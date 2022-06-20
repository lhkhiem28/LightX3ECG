
import os, sys
class config():
    def __init__(self, 
        is_multilabel = False, 
        num_gpus = 1, 
    ):
        self.ecg_leads = [
            0, 1, 
            6
        ]
        self.ecg_length = 5024

        self.is_multilabel = is_multilabel
        self.device_ids = list(range(num_gpus))