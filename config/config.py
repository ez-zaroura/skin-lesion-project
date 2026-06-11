from pathlib import Path
import torch

# Define the base directories for the project
class CFG:
    #paths
    data_dir    = Path("data/isic2020")
    train_csv   = data_dir / "train.csv"
    train_imgs  = data_dir / "train"
    output_dir  = Path("outputs")
    
    #model
    model_name  = "efficientnet_b4"
    pretrained  = True
    num_classes = 1
    img_size    = 380
    
    #training
    epochs      = 10
    batch_size  = 32
    lr         = 1e-4
    weight_decay = 1e-5
    seed       = 42
    n_folds    = 3
    num_workers = 2

    #Focal loss
    focal_alpha = 0.25
    focal_gamma = 2.0

    #Other
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp     = True 
    tta_steps   = 5 
