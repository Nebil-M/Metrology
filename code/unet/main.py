
from unet_inference import infer_all

def run(): 
    # train_stage(
    #         data_root = "/Users/alexstrugacz/ml-research/Metrology/data/training1",   # has /image/*.tif and /label/*.tif
    #         model_output = "/Users/alexstrugacz/ml-research/Metrology/models/Alex_first.pth",
    #         epochs=50, lr=1e-4, bs=8,
    #         device="cpu", channels=(32,64,128,256))
        """ Run U-Net inference with specified model path and data paths inside the function."""
        infer_all(
            model_pth="/Users/alexstrugacz/ml-research/Metrology/models/Double_Unet_Nebil.pth",
            images_root="/Users/alexstrugacz/ml-research/Metrology/data/training1/image",
            save_root="/Users/alexstrugacz/ml-research/Metrology/data/infer",
            device="cpu"
        )



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    run()