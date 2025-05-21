import torch
import multiprocessing as mp

from multiprocessing import Process

from utils.methods import run_method

from algorithms.lora import LoraAlgorithm
from algorithms.lisa import LisaAlgorithm
from algorithms.vaccine import VaccineAlgorithm
from algorithms.vaccine_lora import VaccineLoraAlgorithm

from finetune_datasets import FinetuneDataSet


datasets = [
    FinetuneDataSet.SST2_BENIGN,
    FinetuneDataSet.SST2_HIGH_HARM
]

methods = [
    VaccineLoraAlgorithm,
    LisaAlgorithm,
    LoraAlgorithm,
]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    for dataset in datasets:
        for method_cls in methods:
            p = Process(target=run_method, args=(method_cls, dataset))
            p.start()
            p.join()
            torch.cuda.empty_cache()