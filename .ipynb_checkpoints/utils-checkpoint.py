import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torch
from datasets import *
from vae import VAE
from liu_vae import VAE_LIU
import time
import argparse
import matplotlib
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_id", default=100)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--batch_size_test", default=8, type=int)
    parser.add_argument("--num_epochs", default=2000, type=int)
    parser.add_argument("--latent_img_size", default=32, type=int)
    parser.add_argument("--z_dim", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--delta", default=1.0, type=float) # new_param
    parser.add_argument("--exp", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--dataset", default="panoptics")
    parser.add_argument("--category", default='all')
    parser.add_argument("--defect", default=None)
    parser.add_argument(
        "--defect_list",
        type=lambda s: [item for item in s.split(',')]
    )

    parser.add_argument("--nb_channels", default=3, type=int)
    parser.add_argument("--force_train", dest='force_train', action='store_true')
    parser.set_defaults(force_train=False)
    parser.add_argument("--force_cpu", dest='force_cpu', action='store_true')
    parser.add_argument("--dst_dir", type=str, default=os.getcwd())
    parser.add_argument("--data_dir", type=str, default="/home/getch/DATA/VAE/data",required=False)
    parser.add_argument("--ndvi", dest="ndvi", action='store_true', default=False)
    parser.set_defaults(ndvi=False)
    parser.add_argument("--anomaly", help='anomaly score map approach, could be either of "ssim" or "ssim_mad" ', type=str, default='ssim')
    parser.add_argument("--blur", dest='blur', help='Whether to blur an image', default=False, action='store_true')
    parser.set_defaults(blur=False)
    parser.add_argument("--scale", help='Scale factor to downscale the image to coarser resolution', type=int, default=16)
    parser.add_argument("--liu_vae", help='Whether to use Liu VAE implementation approach', dest='liu_vae', action='store_true')
    parser.set_defaults(liu_vae=False)  # conv_layers
    
    parser.add_argument("--disc_module", help='Whether to use disc_module for liu_vae', dest='disc_module', action='store_true')
    parser.set_defaults(disc_module=False)  # conv_layers
    
    
    parser.add_argument('--conv_layers', type=lambda s: re.split(',', s), help='list of convlayers for liu_vae implementation attanetion map generation', required=False, default="conv_1,conv_2")


    return parser.parse_args()

def load_vae(args):
    if not args.liu_vae:
        model = VAE(
            latent_img_size=args.latent_img_size,
            z_dim=args.z_dim,
            img_size=args.img_size,
            nb_channels=args.nb_channels,
            beta=args.beta,)
    else:
        model = VAE_LIU(
            latent_img_size=args.latent_img_size,
            z_dim=args.z_dim,
            img_size=args.img_size,
            nb_channels=args.nb_channels,
            beta=args.beta,
            delta=args.delta,
            liu_vae=args.liu_vae,
            disc_module=False)

    return model

def load_model_parameters(model, file_name, dir1, dir2, device):
    print(f"Trying to load: {file_name}")
    try:
        state_dict = torch.load(
            os.path.join(dir1, file_name),
            map_location=device
        )
    except FileNotFoundError:
        state_dict = torch.load(
            os.path.join(dir2, file_name),
            map_location=device
        )
    model.load_state_dict(state_dict, strict=False)
    print(f"{file_name} loaded !")

    return model

def get_train_dataloader(args):
    if args.dataset == "panoptics":
        train_dataset = PanopticsTrainDataset(
            args.category,
            args.img_size,
        )
    else:
        raise RuntimeError("No / Wrong dataset provided")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False if args.dataset == "ssl_vqvae" else True,
        num_workers=12
    )

    return train_dataloader, train_dataset

def get_test_dataloader(args, with_loc=False, categ='all',
    fake_dataset_size=None): # categ=None is added
    if args.dataset == "panoptics":
        test_dataset = PanopticsTestDataset(
            args.img_size,
            category=categ,
            fake_dataset_size=fake_dataset_size
        )
    else:
        raise RuntimeError("No / Wrong dataset provided")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        num_workers=12
    )

    return test_dataloader, test_dataset

def tensor_img_to_01(t, share_B=False):
    ''' t is a BxCxHxW tensor, put its values in [0, 1] for each batch element
    if share_B is False otherwise normalization include all batch elements
    '''
    t = torch.nan_to_num(t)
    if share_B:
        t = ((t - torch.amin(t, dim=(0, 1, 2, 3), keepdim=True)) /
            (torch.amax(t, dim=(0, 1, 2, 3), keepdim=True) - torch.amin(t,
            dim=(0, 1, 2,3),
            keepdim=True)))
    if not share_B:
        t = ((t - torch.amin(t, dim=(1, 2, 3), keepdim=True)) /
            (torch.amax(t, dim=(1, 2, 3), keepdim=True) - torch.amin(t, dim=(1, 2,3),
            keepdim=True)))
    return t

def update_loss_dict(ld_old, ld_new):
    for k, v in ld_new.items():
        if k in ld_old:
            ld_old[k] += v
        else:
            ld_old[k] = v
    return ld_old

def print_loss_logs(f_name, out_dir, loss_dict, epoch, exp_name):
    if epoch == 0:
        with open(f_name, "w") as f:
            print("epoch,", end="", file=f)
            for k, v in loss_dict.items():
                print(f"{k},", end="", file=f)
            print("\n", end="", file=f)
    # then, at every epoch
    with open(f_name, "a") as f:
        print(f"{epoch + 1},", end="", file=f)
        for k, v in loss_dict.items():
            print(f"{v},", end="", file=f)
        print("\n", end="", file=f)
    if (epoch + 1) % 50 == 0 or epoch in [4, 9, 24]:
        # with this delimiter one spare column will be detected
        arr = np.genfromtxt(f_name, names=True, delimiter=",")
        fig, axis = plt.subplots(1)
        for i, col in enumerate(arr.dtype.names[1:-1]):
            axis.plot(arr[arr.dtype.names[0]], arr[col], label=col)
        axis.legend()
        fig.savefig(os.path.join(out_dir,
            f"{exp_name}_loss_{epoch + 1}.png"))
        plt.close(fig) 
