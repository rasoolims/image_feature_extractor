import os
from optparse import OptionParser

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import transforms

import dataset
import image_model


def get_options():
    parser = OptionParser()
    parser.add_option("--data", dest="data_path", help="Path to the data folder", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_path", help="Path to the output file", metavar="FILE", default=None)
    parser.add_option("--batch", dest="batch", help="Batch size", type="int", default=128)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    data = dataset.ImageTextDataset(data_folder=options.data_path, transform=transform)
    data_size = len(data.image_paths)
    loader = data_utils.DataLoader(data, batch_size=options.batch, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = image_model.init_net().to(device)

    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    done_work = 0
    for i, batch in enumerate(loader):
        with torch.no_grad():
            images = batch["images"].to(device)
            paths = batch["paths"]
            grid, final = model(images)
            grid = grid.numpy()
            final = final.numpy()
            for p, path in enumerate(paths):
                dir_name = os.path.dirname(path)
                f_name = os.path.basename(path)
                f_name = f_name[:f_name.rfind(".")] + ".npz"
                out_dir_name = os.path.join(options.output_path, dir_name)
                if not os.path.exists(out_dir_name):
                    os.makedirs(out_dir_name)

                np.savez_compressed(os.path.join(out_dir_name, f_name), grid=grid[p], final=final[p])

            done_work += len(paths)
            print("done", done_work, "out of", data_size)
