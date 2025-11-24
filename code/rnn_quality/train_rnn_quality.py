import os
import sys

from training import train_quality_stage
from inference import predict_pair

class QualityRNNManager:

    @staticmethod
    def run_train(data_root, model_output):
        train_quality_stage(
            data_root=data_root,
            model_output=model_output,
            epochs=30,     
            lr=1e-4,
            bs=8,
            device="cuda",  
            base_channels=16,
            rnn_hidden=64,
            bidirectional=False
        )


def run():
    # POINT THIS TO THE NEW FOLDER WE BUILT
    data_root = r"C:/Repo/Metrology/Evaluator_Dataset_balanced" 
    
    model_output = r"C:/Repo/Metrology/models/rnn_quality_v4.pth"
    QualityRNNManager.run_train(
        data_root=data_root,
        model_output=model_output,
    )

if __name__ == "__main__":
    import multiprocessing as mp
    # Ensure safe multiprocessing on Windows
    mp.set_start_method("spawn", force=True)
    run()