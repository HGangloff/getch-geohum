#!/bin/bash

python vae_train.py\
    --exp=vae_vae\
    --dataset=panoptics\
    --category=Minawao_june_2016\
    --num_epochs=100\
    --lr=1e-4\
    --img_size=256\
    --batch_size=16\
    --batch_size_test=8\
    --latent_img_size=32\
    --z_dim=256\
    --beta=0.1\
    --delta=1.0\
    --nb_channels=4\
    --force_train\
    --dst_dir=D:/ANOMALY_liu\
    --anomaly=ssim\
    --scale=16\
    --blur\
    --liu_vae\
    #--disc_module\
    #--conv_layers="conv_1,conv_2,conv_3"\
    
    #--category=Deghale_Apr_2017\
    #--category=Kule_tirkidi_jun_2018\
    #--category=Kule_tirkidi_marc_2017\
    #--category=Kutuplong_dec_2017\
    #--category=Minawao_feb_2017\
    #--category=Minawao_june_2016\
    #--category=Nguyen_march_2017\
    #--category=Tza_oct_2016\
    #--category=Zamzam_april_2022\
