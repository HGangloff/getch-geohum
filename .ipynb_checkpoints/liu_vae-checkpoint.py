import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import resnet18

class VAE_LIU(nn.Module):

    def __init__(self, img_size, nb_channels, latent_img_size, z_dim,
        rec_loss="xent", beta=1, delta=1, liu_vae=False, disc_module=False):
        '''
        liu_vae constructs the Liu 2020 VAE in Towards explainable AD
        it consists of resnet18 in the encoder then Linear(1024) before the
        Linear bottleneck with z_dim units and the symmetrical decoder

        if not liu_vae, then we have the resnet18 which can be followed by more
        convolutions until we find an image of size latent_img_size which will
        be the dimensions of our latent space which is convolutional in this
        case
        '''
        super(VAE_LIU, self).__init__()

        # we need power of 2
        assert (img_size & (img_size - 1) == 0) 
        assert (latent_img_size & (latent_img_size - 1) == 0) 

        if liu_vae:
            assert img_size == 256

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size # size of the latent "image" bc z
        # convolutional
        if liu_vae:
            self.z_dim = z_dim #32
        else:
            self.z_dim = z_dim # depth of the latent image
        self.rec_loss = rec_loss
        self.beta = beta
        self.delta = delta
        self.disc_module = disc_module
        self.liu_vae = liu_vae

        # find the number of convolutions needed in the encoder to go from
        # img_size to the latent_img_size given a convolution divides the
        # size by 2
        if self.liu_vae:
            #self.nb_conv = 5
            self.nb_conv = int(np.log2(img_size // latent_img_size))
        else:
            self.nb_conv = int(np.log2(img_size // latent_img_size))
        # the depth we will have at the end of the encoder given that a
        # convolution incease depth by 2 starting at 32 after the first
        self.max_depth_conv = 2 ** (4 + self.nb_conv)
        
        # NOTE we cannot declare nn layers in the encoder or decoder function
        # if we want the parameters to be on the good device they need to be
        # created here
        # construct the encoder parameters here
        if self.liu_vae:
            self.resnet = resnet18(pretrained=False)
        else:
            self.resnet = resnet18(pretrained=False)
        self.resnet_entry = nn.Sequential(
            # need to rewrite self.resnet.conv1 to handle single channel
            # depth_in: self.nb_channels, depth_out: 32
            # NOTE why 64 here ?
            nn.Conv2d(self.nb_channels, 64, kernel_size=7,
                stride=2, padding=3, bias=False),
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.resnet18_layer_list = [
            self.resnet.layer1, # depth_in: 64, depth_out: 64
            self.resnet.layer2, # depth_in: 64, depth_out: 128
            self.resnet.layer3, # depth_in: 128, depth_out: 256
            self.resnet.layer4  # depth_in: 256, depth_out: 512
        ]
        self.encoder_layers = [self.resnet_entry] # the first is mandatory
        for i in range(1, self.nb_conv): # -1 because we already have 1 layer
            try:
                self.encoder_layers.append(self.resnet18_layer_list[i - 1])
            except IndexError: # if we have used all the reset net layers
                depth_in = 2 ** (4 + i)
                depth_out = 2 ** (4 + i + 1)
                self.encoder_layers.append(nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, 4, 2, 1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU()
                    ))
        self.conv_encoder = nn.Sequential(
            *self.encoder_layers,
        )
        if self.liu_vae:
            self.final_encoder = nn.Sequential(
                #nn.Linear(512 * 8 * 8, 1024),   # the two final layers from Liu
                #nn.Linear(1024, self.z_dim * 2) # self.z_dim is 32
                #nn.Conv2d(512, self.z_dim * 2, kernel_size=1,
                #stride=1, padding=0)
                nn.Conv2d(self.max_depth_conv, self.z_dim * 2, kernel_size=1,
                stride=1, padding=0)
            )
        else:
            # the final conv2D to get a convolutional z with the right depth
            # not appended to conv_encoder because it will be different for VQVAE
            self.final_encoder = nn.Sequential(
                nn.Conv2d(self.max_depth_conv, self.z_dim * 2, kernel_size=1,
                stride=1, padding=0)
            )

        if self.liu_vae:
            self.initial_decoder = nn.Sequential(
                #nn.Linear(self.z_dim, 1024),
                #nn.Linear(1024, 1024 * 4 * 4),
                #nn.ConvTranspose2d(self.z_dim, 512,
                #    kernel_size=1, stride=1, padding=0),
                nn.ConvTranspose2d(self.z_dim, self.max_depth_conv,
                    kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.max_depth_conv),
                nn.ReLU()
            )
            
        else:
            # construct the decoder parameters
            # here the initial conv that align the depth is already appended
            # because it is the same in VQVAE
            self.initial_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.z_dim, self.max_depth_conv,
                    kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.max_depth_conv),
                nn.ReLU()
            )
            
        if self.liu_vae:
            # nb_conv_dec = self.nb_conv + 1 # because we force conv z in Liu
            nb_conv_dec = self.nb_conv #+ 1
        else:
            nb_conv_dec = self.nb_conv

        self.decoder_layers = []
        for i in reversed(range(nb_conv_dec)):
            depth_in = 2 ** (4 + i + 1)
            depth_out = 2 ** (4 + i)
            if i == 0:
                depth_out = self.nb_channels
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                ))
            else:
                #if self.liu_vae and i == nb_conv_dec - 1:
                #    self.decoder_layers.append(nn.Sequential(
                #        nn.ConvTranspose2d(1024, depth_out, 4, 2, 1),
                #        nn.BatchNorm2d(depth_out),
                #        nn.ReLU()
                #    ))
                #else:
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU()
                ))
        self.conv_decoder = nn.Sequential(
            *self.decoder_layers
        )

        # some addons now 

        # placeholder for the activation gradients we want to access
        # in Liu gradcam
        self.gradients = {}

        # discriminator
        self.size_last_conv_disc = img_size // (2 ** 4) # bc 4 conv2D currently
        self.conv_disc = nn.Sequential(
            nn.Conv2d(self.nb_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * self.size_last_conv_disc ** 2, 1)
        )

    def activation_hook(self, grad, conv_id):
        '''
        the hook on tensor  must either return None or a Tensor
        which will be used in place of grad for further gradient computation.
        We provide an example below.
        grad is the mandatory only element for a hook function normally but
        here we use a wrapper
        '''
        self.gradients[conv_id] = grad

    def get_activation_gradient(self, conv_id):
        return self.gradients[conv_id]

    def get_activations(self, x, conv_id):
        ''' get the activation (feature map) at the selected convolution of the
            encoder '''
        x = self.resnet_entry(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        if conv_id == 'conv_1':
            return x1
        elif conv_id == 'conv_2':
            return x2
        elif conv_id == 'conv_3':
            return x3
        else:
            return x4

    def encoder(self, x):
        if self.liu_vae:
            x = self.resnet_entry(x)
            x1 = self.resnet.layer1(x)
            _ = x1.register_hook(lambda grad: self.activation_hook(grad, 'conv_1'))
            x2 = self.resnet.layer2(x1)
            _ = x2.register_hook(lambda grad: self.activation_hook(grad, 'conv_2'))
            x = x2
            #x3 = self.resnet.layer3(x2)
            #_ = x3.register_hook(lambda grad: self.activation_hook(grad, 'conv_3'))
            #x4 = self.resnet.layer4(x3)
            #_ = x4.register_hook(lambda grad: self.activation_hook(grad, 'conv_4'))
            # now we stop resnet because we do resnet without classif
            #x = x4.view(x4.shape[0], -1)
            #x = x4 # because we force a convolutional latent space for Liu
        else:
            x = self.conv_encoder(x)

        x = self.final_encoder(x)
        return x[:, :self.z_dim], x[:, self.z_dim:]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decoder(self, z):
        #if self.liu_vae:
        #    z = z.view(-1, self.z_dim)
        z = self.initial_decoder(z)
        #if self.liu_vae:
        #    z = z.view(-1, 1024, 4, 4)
        x = self.conv_decoder(z)
        x = nn.Sigmoid()(x)
        return x

    def discriminator(self, x):
        x = self.conv_disc(x)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return self.decoder(z), (mu, logvar)

    def mse(self, recon_x, x):
        return torch.sum(torch.square(recon_x - x), dim=(1, 2, 3))

    def xent_continuous_ber(self, recon_x, x, pixelwise=False):
        ''' p(x_i|z_i) a continuous bernoulli '''
        eps = 1e-6
        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps) # bc infinity is not cool
            #x = (x < eps) * eps + (x >= 1 - eps) * (1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
                    torch.ones_like(x))
            #x = (x > 0.49 & x < 0.51) * 0.49 + (x < 0.49 | x > 0.51) * x
            #return torch.log((2 * torch.arctanh(1 - 2 * x)) /
            return torch.log((2 * self.tarctanh(1 - 2 * x)) /
                            (1 - 2 * x) + eps)
        if pixelwise:
            return (x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x))
        else:
            # NOTE with the following summation in log space, it is then transparent to
            # have more than one channel
            return torch.sum(x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x), dim=(1, 2, 3))
        # NOTE we take the mean over the image size as everybody does for the
        # hyperparameters to be like the other. But this averagin over the
        # image size is not mathematically imposed in the formula !!!
        # NOTE NOTE NOTE
        #return torch.mean(x * torch.log(recon_x + eps) +
        #                (1 - x) * torch.log(1 - recon_x + eps) +
        #                log_norm_const(recon_x), dim=(1, 2, 3))

    def mean_from_lambda(self, l):
        ''' because the mean of a continuous bernoulli is not its parameter '''
        l = torch.clamp(l, 10e-6, 1 - 10e-6) # bc infinity is not cool
        #l = (l < eps) * eps + (l >= 1 - eps) * (1 - eps)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 *
            torch.ones_like(l))
        #l = (l > 0.49 & l < 0.51) * 0.49 + (l < 0.49 | l > 0.51) * l
        #return l / (2 * l - 1) + 1 / (2 * torch.arctanh(1 - 2 * l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

    def dkl(self):
        # NOTE -kld actually
        return 0.5 * torch.sum(
                1 + self.logvar - self.mu.pow(2) - self.logvar.exp(),
            dim=(1)
        )

    def loss_function(self, recon_x, x):
        if self.rec_loss == "xent":
            rec_term = self.xent_continuous_ber(recon_x, x)
        elif self.rec_loss == "mse":
            rec_term = -self.mse(recon_x, x)
        rec_term = torch.mean(rec_term)

        kld = torch.mean(self.dkl())

        L = (rec_term + self.beta * kld)
        if self.disc_module:
            L_adv = torch.mean(
                torch.log(self.discriminator(x)) + torch.log(1 -
                    self.discriminator(recon_x))
                )
        else:
            L_adv = 0

        loss = L + self.delta * L_adv

        loss_dict = {
            'loss': loss,
            'rec_term': rec_term,
            '-beta*kld': self.beta * kld
        }
        if self.disc_module:
            loss_dict['delta*l_adv'] = self.delta * L_adv

        return loss, loss_dict

    def step(self, input_mb):
        recon_mb, _ = self.forward(input_mb)

        loss, loss_dict = self.loss_function(recon_mb, input_mb)

        if self.rec_loss == "xent":
            # NOTE do this after the loss function
            recon_mb = self.mean_from_lambda(recon_mb)

        return loss, recon_mb, loss_dict

    def tarctanh(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

        
