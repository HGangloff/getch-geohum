import torch

def liu_anomaly_map(conv_id=None, model=None,imgs=None, device=None):
    # compute the gradient wrt the model parameters for each latent z
    # and the intermediate gradient wrt to the desired activation map will be
    # saved since we have set up a hook
    A = model.get_activations(imgs, conv_id).detach()
    alphas = torch.zeros((A.shape[1])).detach().to(device)
    M = torch.zeros((1, A.shape[-2], A.shape[-1])).to(device)
    for k in range(M.shape[0]):
        x_rec, (mu, logvar) = model.forward(imgs)
        if model.rec_loss == "xent":
            x_rec = model.mean_from_lambda(x_rec)
        #mu_i = mu[0, k] #torch.sum(mu)#, dim=(0, -2,-1))[k]
        mu_i = torch.sum(mu)#, dim=(0, -2,-1))[k]
        A = model.get_activations(imgs, conv_id).detach()
        model.zero_grad() # really needed right ?
        mu_i.backward(retain_graph=True) 
        # get the gradient of the activation map (computed wrt z as we have just
        # done)
        act_grad = model.get_activation_gradient(conv_id)
        alphas = torch.mean(act_grad, dim=(0, 2, 3)).detach()

        #M_ = torch.mean(A,dim=1)
        #M_ = torch.mean(act_grad,dim=1)
        M_ = torch.sum(alphas[None, :, None, None] * A, dim=1)
        #M[k] = torch.abs(M_)
        M[k] = torch.nn.ReLU()(M_)
    # suppress border effects ?
    M[:, :2, :]  = 0 #torch.mean(M[0, 2:-2, 2:-2])
    M[:, :, -2:] = 0 #torch.mean(M[0, 2:-2, 2:-2]) 
    M[:, :, :2]  = 0 #torch.mean(M[0, 2:-2, 2:-2])
    M[:, -2:, :] = 0 #torch.mean(M[0, 2:-2, 2:-2])

    M = torch.mean(M, dim=0)
    #M = transforms.functional.resize(M[:, None], (256, 256))
    if conv_id == 'conv_1':
        M = torch.repeat_interleave(M, 4, dim=-2)
        M = torch.repeat_interleave(M, 4, dim=-1)
    elif conv_id == 'conv_2':
        M = torch.repeat_interleave(M, 8, dim=-2)
        M = torch.repeat_interleave(M, 8, dim=-1)
    elif conv_id == 'conv_3':
        M = torch.repeat_interleave(M, 16, dim=-2)
        M = torch.repeat_interleave(M, 16, dim=-1)
    elif conv_id == 'conv_4':
        M = torch.repeat_interleave(M, 32, dim=-2)
        M = torch.repeat_interleave(M, 32, dim=-1)


    return M, x_rec.detach()
