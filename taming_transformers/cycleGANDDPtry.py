import argparse
import math
import random
import io
import os, sys
sys.path.append("./taming_transformers")
import numpy as np
from tqdm import tqdm
import json
from taming.models.vqgan import VQModel
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.utils as vutils
import itertools

from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import transforms
from PIL import Image
import pandas as pd
from omegaconf import OmegaConf


# Define the dataset class
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # self.csv_read = pd.read_csv(os.path.join(root_dir,csv_file))
        self.csv_read = pd.read_csv(csv_file, na_values=[''])
        self.image_paths = self.csv_read['image_path'].values
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        row = self.csv_read.iloc[idx].to_dict()
        
        return image, row

class CycleGAN:
    def __init__(self, config_path, ckpt_path, num_of_new_generated_img = 60, G_lr=2e-5, D_lr=1e-4, beta1=0.9, beta2=0.999, lambda_cycle= 1.0, device='cuda'):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_cycle = lambda_cycle
        self.device = device
        self.G_XtoY = None
        self.G_YtoX = None
        self.D_X = None
        self.D_Y = None
        self.optimizer_G = None
        self.optimizer_D_X = None
        self.optimizer_D_Y = None
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.start_iter = 0
        self.imsave_path = None
        self.model_path = None
        self.num_of_new_generated_img = num_of_new_generated_img

        self.local_rank = int(os.environ['LOCAL_RANK'])

    
    def load_config(self, config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config

    def load_vqgan(self, config, ckpt_path=None, is_gumbel=False):
        
        # model = VQModel(**config.model.params)
        model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False) 

        return model

    def build_models(self):
        config = self.load_config(self.config_path, display=False)
        vqgan = self.load_vqgan(config, ckpt_path=self.ckpt_path).to(self.device)
        self.G_XtoY = vqgan.to(self.device)
        self.G_YtoX = vqgan.to(self.device)
        self.D_X = vqgan.loss.to(self.device)
        self.D_Y = vqgan.loss.to(self.device)
        self.G_XtoY =  DDP(self.G_XtoY, device_ids=[self.local_rank], broadcast_buffers=False)
        self.G_YtoX =  DDP(self.G_YtoX, device_ids=[self.local_rank], broadcast_buffers=False)
        self.D_X =  DDP(self.D_X , device_ids=[self.local_rank], broadcast_buffers=False)
        self.D_Y =  DDP(self.D_Y, device_ids=[self.local_rank], broadcast_buffers=False)
    
    def delete_models(self):
        del self.G_XtoY, self.G_YtoX, self.D_X, self.D_Y   

    def build_optimizers(self):
        self.optimizer_G = optim.Adam(itertools.chain(filter(lambda p: p.requires_grad and p is not self.G_XtoY.loss.parameters(), self.G_XtoY.parameters()), 
            filter(lambda p: p.requires_grad  and p is not self.G_YtoX.loss.parameters(), self.G_YtoX.parameters())),
            lr=self.G_lr,
            betas = (self.beta1, self.beta2)
        )
        self.optimizer_D = optim.Adam(itertools.chain(
            self.D_X.parameters(),self.D_Y.parameters()),
            lr=self.D_lr,
            weight_decay=0.001,
        )

    def set_paths(self, imsave_path, model_path):
        self.imsave_path = imsave_path
        self.model_path = model_path
        if not os.path.exists(imsave_path):
            os.makedirs(imsave_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def save_model(self, iteration):
        torch.save({
            'start_iter': self.start_iter,
            'G_XtoY': self.G_XtoY.state_dict(),
            'G_YtoX': self.G_YtoX.state_dict(),
            'D_X': self.D_X.state_dict(),
            'D_Y': self.D_Y.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_X': self.optimizer_D_X.state_dict(),
            'optimizer_D_Y': self.optimizer_D_Y.state_dict()
        }, os.path.join(self.model_path, f'cycleGAN_{iteration}.pt'))
      
    def sample_data(self, loader):
        while True:
            for batch in loader:
                yield batch
    
    def load_datasets(self, data_root, X_csv, Y_csv, batch_size):

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # Normalize to [-1,1]
])
       
        X_dataset = ImageDataset(csv_file= X_csv, root_dir=data_root, transform=transform)
        Y_dataset = ImageDataset(csv_file= Y_csv, root_dir=data_root, transform=transform)

        samplerX = torch.utils.data.distributed.DistributedSampler(X_dataset)
        samplerY = torch.utils.data.distributed.DistributedSampler(Y_dataset)
    
        self.dataloader_X =  DataLoader(X_dataset, batch_size=batch_size, shuffle=False, sampler=samplerX)
        self.dataloader_Y =  DataLoader(Y_dataset, batch_size=batch_size, shuffle=False, sampler=samplerY)
        self.batch_size = batch_size
        self.Y_csv = Y_csv
        self.X_csv = X_csv
        self.data_root = data_root

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss


    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
        return d_loss

    def train(self, iterations):
        # train the CycleGAN model
        self.build_models()
        self.build_optimizers()
        # self.G_XtoY = nn.DataParallel(self.G_XtoY)
        # self.G_YtoX = nn.DataParallel(self.G_YtoX)
        # self.D_X = nn.DataParallel(self.D_X)
        # self.D_Y = nn.DataParallel(self.D_Y)
        self.G_XtoY.train()
        self.G_YtoX.train()
        self.D_X.train()
        self.D_Y.train()

        loader_X = self.sample_data(self.dataloader_X)
        loader_Y = self.sample_data(self.dataloader_Y)

        # Define the loss function
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()

        pbar = range(iterations)
        pbar = tqdm(pbar, initial=0,
                    dynamic_ncols=True, smoothing=0.01)

        # balancing_lambda = 1/len(self.dataloader_X.dataset.image_paths)     
        balancing_lambda = 1/(iterations*self.batch_size)**(1/3)  
        balancing_lambda2 = 1. # (1/15)**(1/3)    

        # initialize loss arrays
        D_X_losses, D_Y_losses, loss_G_X_all, loss_G_Y_all, cycle_loss_X_all, cycle_loss_Y_all = [], [], [], [], [], []
        qant_lambda = 1.

        for idx in pbar:
            if idx > iterations:
                print("Done!")
                break
            real_img_X, row = next(loader_X)
            real_img_X = real_img_X.to(self.device)

            real_img_Y, row = next(loader_Y)
            real_img_Y = real_img_Y.to(self.device)

            # generate fake images
            fake_img_Y, latent_codes, vq_loss_x  = self.G_XtoY(real_img_X)
            reconstructed_img_X, latent_codes, vq_loss_x_re  = self.G_YtoX(fake_img_Y)

            fake_img_X, latent_codes, vq_loss_y  = self.G_YtoX(real_img_Y)
            reconstructed_img_Y, latent_codes, vq_loss_y_re  = self.G_XtoY(fake_img_X)

            # train the generators
            self.set_requires_grad([self.D_X, self.D_Y], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()
            pred_fake_X = self.D_X(fake_img_X)
            loss_G_X = -torch.mean(pred_fake_X)* balancing_lambda
            pred_fake_Y = self.D_Y(fake_img_Y)
            loss_G_Y =  -torch.mean(pred_fake_Y) * balancing_lambda2
            
            loss_G_X_all.append(loss_G_X.item())
            loss_G_Y_all.append(loss_G_Y.item())

            # calculate cycle consistency loss
            cycle_loss_X = self.criterion_cycle(reconstructed_img_X, real_img_X) * self.lambda_cycle * balancing_lambda
            cycle_loss_Y = self.criterion_cycle(reconstructed_img_Y, real_img_Y) * self.lambda_cycle * balancing_lambda2

            loss_cycle = cycle_loss_X + cycle_loss_Y 
            cycle_loss_X_all.append(cycle_loss_X.item())
            cycle_loss_Y_all.append(cycle_loss_Y.item())
            loss_G = loss_G_X + loss_G_Y + cycle_loss_X + cycle_loss_Y + vq_loss_x * balancing_lambda * qant_lambda + vq_loss_x_re* balancing_lambda  * qant_lambda + vq_loss_y* balancing_lambda2 * qant_lambda + vq_loss_y_re* balancing_lambda2 * qant_lambda
            loss_G.backward()
            self.optimizer_G.step()
            
            # train the discriminators
            self.set_requires_grad([self.D_X, self.D_Y], True)  
            self.optimizer_D.zero_grad()
            pred_real_X = self.D_X(real_img_X)
            pred_fake_X = self.D_X(fake_img_X.detach())
            
            D_X_loss = self.hinge_d_loss(pred_real_X,pred_fake_X) * balancing_lambda
            D_X_loss.backward()
            D_X_losses.append(D_X_loss.item())

            pred_real_Y = self.D_Y(real_img_Y)
            pred_fake_Y = self.D_Y(fake_img_Y.detach())
            
            D_Y_loss = self.hinge_d_loss(pred_real_Y, pred_fake_Y) * balancing_lambda2
            D_Y_loss.backward()
            self.optimizer_D.step()
            D_Y_losses.append(D_Y_loss.item())   
                
            

        # plot and save losses
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        axs[0, 0].plot(D_X_losses)
        axs[0, 0].set_title("D_X Loss")
        axs[0, 1].plot(D_Y_losses)
        axs[0, 1].set_title("D_Y Loss")
        axs[1, 0].plot(loss_G_X_all)
        axs[1, 0].set_title("G XtoY Loss")
        axs[1, 1].plot(loss_G_Y_all)
        axs[1, 1].set_title("G YtoX Loss")
        axs[2, 0].plot(cycle_loss_X_all)
        axs[2, 0].set_title("cycle_loss_X")
        axs[2, 1].plot(cycle_loss_Y_all)
        axs[2, 1].set_title("cycle_loss_Y")
        
        with torch.set_grad_enabled(False):
            self.G_XtoY.eval()
            self.G_YtoX.eval()

            save_dir = self.data_root+'/temp/'            
            # csv_pd = pd.read_csv(os.path.join(self.data_root , self.Y_csv))
            csv_pd = pd.read_csv(self.Y_csv, na_values=[''])
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)


            

            import datetime

            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            cp_save_dir = f'{self.data_root}/temp/{current_time}_lambada_{self.lambda_cycle}/'

            if not os.path.exists(cp_save_dir):
                os.makedirs(cp_save_dir)
            plt.savefig(cp_save_dir+"/losses.png")            
            gen_count = 0
            box_dict = {}
            with open('/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train.json', 'r') as f:
                box_dict_train = json.load(f)

            for index, row in csv_pd.iterrows():
                if not box_dict_train.get(row['filename']):
                    csv_pd = csv_pd.drop(index)



            while gen_count < self.num_of_new_generated_img:
                # Generate fake images from test data
                real_img_X, row = next(loader_X)
                real_img_X = real_img_X.to(self.device)
                fake_Y, latent_codes, _ = self.G_XtoY(real_img_X)
                real_img_Y, _ = next(loader_Y)
                real_img_Y = real_img_Y.to(self.device)
                fake_X, latent_codes, _ = self.G_YtoX(real_img_Y)

                reconstructed_img_X, latent_codes, _  = self.G_YtoX(fake_Y)
                reconstructed_img_Y, latent_codes, _  = self.G_XtoY(fake_X)
                
                new_row = {}
                

                keys = list(row.keys())
                for key in keys:
                    new_row[key] = []

                sample = torch.cat((real_img_X, fake_Y, reconstructed_img_X, real_img_Y, fake_X, reconstructed_img_Y))

                vutils.save_image(
                    sample,
                    f"%s/{str(random.randint(0, 100000)).zfill(6)}_.png" % (cp_save_dir),
                    nrow=int(self.batch_size),
                    normalize=True,
                    range=(-1, 1),
                )
                for i in range(fake_Y.shape[0]):
                    img_name = f"generated_image_{random.randint(0, 100000)}.jpg"
                    gen_img_path = os.path.join(save_dir, img_name)
                    
                    if row['agent1_name'][i] == 'blank' or  np.isnan(row['height'][i]) or np.isnan(row['width'][i]) or int(row['width'][i]) == 0 or int(row['height'][i]) == 0:
                        continue 
                    else:
                        new_height = int(row['height'][i]) #if not np.isnan(row['image_height'][i]) else int(row['image_width'][i])
                        new_width =  int(row['width'][i]) #if not np.isnan(row['image_width'][i]) else int(row['image_height'][i])# New dimensions to resize to
                    
                    resized = torch.nn.functional.interpolate(fake_Y[i].unsqueeze(0), mode='bicubic', size=(new_height, new_width), align_corners=False)#.clamp(min=0, max=255) #, mode='bicubic' , align_corners=False
                    resized = resized.squeeze(0)
                    # resized = fake_Y[i]
                    vutils.save_image(resized, gen_img_path, normalize=True,range=(-1, 1))
                    row['image_path'][i] = 'temp/' + img_name #[12:]
                    # import ipdb; ipdb.set_trace()
                    box_dict[row['filename'][i]] = box_dict_train[row['filename'][i]]
                    box_dict[img_name] = box_dict_train[row['filename'][i]]
                    row['filename'][i] = img_name
                    gen_count += 1

                    for key, value in row.items():
                        if key in keys:
                            if value[i] != '':
                                new_row[key].append(value[i])
                            else:
                                new_row[key].append('')  # Append an empty string when value[i] 
                    # new_row['novel'].append('1')
                # iterate over the new rows and append each row to the existing dataframe
                for i in range(len(new_row['image_path'])):
                    row_dict = {key: new_row[key][i].numpy() if isinstance(new_row[key][i], torch.Tensor) else new_row[key][i] for key in new_row.keys()}
                    csv_pd = csv_pd.append(row_dict, ignore_index=True)
             
            # csv_pd.to_csv(os.path.join(self.data_root, self.Y_csv), index=False)
            csv_pd.to_csv(self.Y_csv, index=False)
            with open(self.Y_csv[:-4]+'.json', 'w') as file:
                # Write the dictionary to the file as json
                json.dump(box_dict, file)
     
                
# cycleGAN = CycleGAN('./taming_transformers/logs/vqgan_imagenet_f16_1024/configs/model.yaml','./taming_transformers/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt')
# cycleGAN.build_models()
# cycleGAN.build_optimizers()
# cycleGAN.load_datasets('./', 'dataset_v4/dataset_v4_2_train.csv', 'dataset_v4/OND.10712.000_single_df.csv', 2)      
# cycleGAN.train(400)

                
# cycleGAN = CycleGAN('./taming_transformers/logs/vqgan_imagenet_f16_1024/configs/model.yaml','./taming_transformers/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt')
# cycleGAN.build_models()
# cycleGAN.build_optimizers()
# cycleGAN.load_datasets('./', 'dataset_v4/dataset_v4_2_train.csv', 'dataset_v4/OND.10713.000_single_df.csv', 1)      
# cycleGAN.train(400)


# cycleGAN = CycleGAN('./taming_transformers/logs/vqgan_imagenet_f16_1024/configs/model.yaml','./taming_transformers/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt')
# cycleGAN.build_models()
# cycleGAN.build_optimizers()
# cycleGAN.load_datasets('./', 'dataset_v4/dataset_v4_2_train.csv', 'session/temp/65924_batch_16_retrain1.csv', 4)      
# cycleGAN.train(1)























