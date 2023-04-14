import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from utils import (get_train_dataloader,
                   get_test_dataloader,
                   load_model_parameters,
                   load_vae,
                   parse_args
                   )
from liu_processing import liu_anomaly_map  # for the model to compare with state of art

def make_normal(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def NDVI(image, channel='first', normalize=True, func=True):
    if normalize:
        if func:
            image = make_normal(image)
        else:
            image = image/1000 # 1000 is image rescaling factor
    if channel=='first':
        r = image[-2, :, :]
        n = image[-1, :, :]
    else:
        r = image[:, :, -2]
        n = image[:, :, -1]
    const = 0.0000001
    ndvi = ((n-r)+const)/((r+n)+const)
    return ndvi

def ssim(a, b, win_size):
    "Structural di-SIMilarity: SSIM"
    a = a.detach().cpu().permute(1, 2, 0).numpy()
    b = b.detach().cpu().permute(1, 2, 0).numpy()

    #b = gaussian_filter(b, sigma=2)

    try:
        score, full = structural_similarity(a, b, #multichannel=True,
            channel_axis=2, full=True, win_size=win_size)
    except ValueError: # different version of scikit img
        score, full = structural_similarity(a, b, multichannel=True,
            channel_axis=2, full=True, win_size=win_size)
    #return 1 - score, np.median(1 - full, axis=2)  # Return disim = (1 - sim)
    return 1 - score, np.product((1 - full), axis=2)

def get_error_pixel_wise(model, x, loss="rec_loss"):
    x_rec, _ = model(x)
    
    return x_rec

def test(args):
    ''' testing pipeline '''
    if args.ndvi:
        print('=====================================================================')
        print('NDVI will be computed as an auxiliary information for post-processing')
        print('=====================================================================')
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Pytorch device:", device)

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    checkpoints_dir =f'{args.dst_dir}/{args.category}/torch_checkpoints'  # "./torch_checkpoints" # '{args.dst_dir}/{args.category}/torch_checkpoints' needs change
    checkpoints_saved_dir =f'{args.dst_dir}/{args.category}/torch_checkpoints_saved' # "./torch_checkpoints_saved"  # need change

    # NOTE force test batch size to be 1
    # NOTE could we process batch of images ?
    args.batch_size_test = 1

    fake_dataset_size=None # means that we process of the dataset, otherwise a
    # integer restricts its size
    auc_file = f"{args.dst_dir}/predictions_blur_{args.blur}_{args.scale}_{args.anomaly}/AUC_{args.anomaly}_ssmsummary.txt"
    if args.category == 'all':
        categories = sorted(os.listdir(args.data_dir))
    else:
        categories = [args.category]
    print(f'Got {len(categories)} folders: \n {categories}')
    for j in range(len(categories)):
        print(f'processing for {categories[j]}')
        predictions_dir =f"{args.dst_dir}/predictions_blur_{args.blur}_{args.scale}_{args.anomaly}/{categories[j]}"   # need change
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir, exist_ok=True)

        test_dataloader, test_dataset = get_test_dataloader(
                args,
                with_loc=True,
                categ=categories[j],
                fake_dataset_size=fake_dataset_size
            )

        # Load model
        model = load_vae(args)
        model.to(device)

        try:
            file_name = f"{args.exp}_{args.params_id}.pth"
            model = load_model_parameters(
                model,
                file_name,
                checkpoints_dir,
                checkpoints_saved_dir, 
                device
            )
        except FileNotFoundError:
            raise RuntimeError("The model checkpoint does not exist !")

        dissimilarity_func = ssim

        classes = {}

        model.eval()

        aucs = []

        pbar = tqdm(test_dataloader)
        for ii,(imgs, gt, imgs_blur) in enumerate(pbar): # changed
            imgs = imgs.to(device)
            imgs_blur = imgs_blur.to(device) # changed

            gt_np = gt[0].cpu().numpy().astype(float)#[..., 0]
            #print('Ground truth shape', gt_np.shape)
            #print('Ground truth dtype', gt_np.dtype)
            #gt_np = (gt_np - np.amin(gt_np)) / (np.amax(gt_np) - np.amin(gt_np))

            ##########################################################
            if args.liu_vae:
                conv_layers = ['conv_1']#, 'conv_2', 'conv_3', 'conv_4']
                anomaly_maps = []
                for c in conv_layers:
                    M, x_rec = liu_anomaly_map(c, model, imgs)
                    anomaly_maps.append(M)

                amaps = np.squeeze(anomaly_maps[0].cpu().numpy())
                pred_score = [np.amax(anomaly_maps[0].cpu().numpy())]
                mask = (amaps > 0.1).astype(np.int8)

            ##########################################################


            x_rec, _ = model(imgs_blur) # changed from imgs
            x_rec = model.mean_from_lambda(x_rec)

            score, ssim_map = dissimilarity_func(x_rec[0],imgs[0], 11)  # later change this to imgs

            ssim_map = ((ssim_map - np.amin(ssim_map)) / (np.amax(ssim_map) - np.amin(ssim_map)))

            # SM metric
            amaps = ssim_map

            amaps = ((amaps - np.amin(amaps)) / (np.amax(amaps) - np.amin(amaps)))
            #print(f'== Shape of amaps: {amaps.shape}==')

            if args.dataset in ["panoptics"]:
                preds = amaps.copy() 
                mask = np.zeros(gt_np.shape)

                try:
                    auc = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds.flatten()) # changed 
                    aucs.append(auc)
                except ValueError:
                    pass
                    # ROCAUC will not be defined when one class only in y_true

            m_aucs = np.mean(aucs)
            pbar.set_description(f"mean ROCAUC: {m_aucs:.3f}")

            ori = imgs[0].permute(1, 2, 0).cpu().numpy()
            if args.ndvi:
                ndvi = NDVI(ori, channel='last') # as channel is converted
            ori = ori[..., :3] # NOTE 4 bands panoptics
            #gt = gt[0].permute(1, 2, 0).cpu().numpy()
            rec = x_rec[0].detach().permute(1, 2, 0).cpu().numpy()
            rec = rec[..., :3] # NOTE 4 bands panoptics
            path_to_save = f'{args.dst_dir}/predictions_blur_{args.blur}_{args.scale}_{args.anomaly}/{categories[j]}/'  # needs reshafling

            if args.ndvi:
                np.save(path_to_save + '{}_ndvi.npy'.format(str(ii)),ndvi) # added 
            img_to_save = Image.fromarray((ori * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_ori.png'.format(str(ii)))       # needs reshafling 
            img_to_save = Image.fromarray(gt_np[0,:,:].astype(np.uint8))   # changed 
            img_to_save.save(path_to_save + '{}_gt.png'.format(str(ii)))    # needs reshafling 
            img_to_save = Image.fromarray((rec * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_rec.png'.format(str(ii)))    # needs reshafling
            np.save(path_to_save + f'{ii}_final_amap.npy', amaps) # new addition to save
            cm = plt.get_cmap('jet')
            amaps = cm(amaps)
            img_to_save = Image.fromarray((amaps[..., :3] * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_final_amap.png'.format(str(ii))) # needs reshafling 

        m_auc = np.mean(aucs)
        with open(auc_file, 'a+') as txt:
            txt.write(f'{categories[j]}: {m_auc}\n')
        print("Mean auc on", categories[j], args.defect, m_auc)


        #return m_auc

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "panoptics":
        test(args)
        #m_auc = test(args)
    else:
        raise RuntimeError("Wrong dataset")
