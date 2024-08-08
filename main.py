from dataloader import get_dataloader

if __name__ == "__main__":

    datapath = "data/massachusetts_split"

    train_loader = get_dataloader(datapath, "train")
    val_loader = get_dataloader(datapath, "val")
    test_loader = get_dataloader(datapath, "test")