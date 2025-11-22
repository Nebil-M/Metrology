# code/train_rnn_quality.py

from Util import QualityRNNManager

def run():
    data_root = r"C:/Repo/Metrology/data/rnn_quality1"          # has image/ and mask/
    labels_csv = r"C:/Repo/Metrology/data/rnn_quality1/labels.csv"
    model_output = r"C:/Repo/Metrology/models/rnn_quality_v1.pth"

    QualityRNNManager.run_train(
        data_root=data_root,
        labels_csv=labels_csv,
        model_output=model_output,
    )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    run()