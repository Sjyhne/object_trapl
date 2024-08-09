from dataloader import get_dataloader

if __name__ == "__main__":

    datapath = "data/massachusetts_object_detection"

    test_loader = get_dataloader(datapath, "val")
    
    for i, (image, bounding_boxes) in enumerate(test_loader):
        print(image.shape)
        print(bounding_boxes)
        break