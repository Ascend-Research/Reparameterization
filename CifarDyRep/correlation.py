import torch
import numpy as np
from fast_histogram import histogram2d

def autocorrelation_to_gradient_multiplier(corr_matrix):
    gradient_multiplier = corr_matrix.clone()
    gradient_multiplier /= torch.mean(gradient_multiplier, dim=(-2,-1), keepdim=True)
    return gradient_multiplier

def shannon_entropy(c):
        c = c[np.nonzero(c)]
        c /= np.sum(c)
        H = -np.sum(c * np.log2(c))  
        return H
   
def NMIK(X, Y, k):
    bins = int(1 + np.ceil(np.log2(len(X))))
    
    range_X = [np.min(X), np.max(X) + 1e-12]
    range_Y = [np.min(Y), np.max(Y) + 1e-12]
    
    c_XY = histogram2d(X, Y, range=[range_X, range_Y], bins=bins)
    c_X = np.sum(c_XY, axis=0)
    c_Y = np.sum(c_XY, axis=1)

    H_X = shannon_entropy(c_X)
    H_Y = shannon_entropy(c_Y)
    H_XY = shannon_entropy(c_XY.flatten())

    MI = H_X + H_Y - H_XY
    NMI = MI/H_XY
    return k * NMI / ((k-1)*NMI + 1)

def feature_map_nmik(input, kernel_size=(3,3), k=1):
    input = input.numpy()
    channels = input.shape[1]
    autocorr_matrices = torch.empty(channels, *kernel_size)
    
    for c in range(channels):
        center_vals = input[:, c, int((kernel_size[0]-1)/2) : input.shape[2] - (kernel_size[0]-int((kernel_size[0]-1)/2)-1), int((kernel_size[1]-1)/2) : input.shape[3] - (kernel_size[1]-int((kernel_size[1]-1)/2)-1)].flatten()
        for x in range(0, kernel_size[0]):
            for y in range(0, kernel_size[1]):

                # center correlation is 1
                if x == int((kernel_size[0]-1)/2) and y == int((kernel_size[1]-1)/2):
                    autocorr_matrices[c,x,y] = 1
                    continue

                neighbor_vals = input[:, c, x : input.shape[2] - (kernel_size[0]-x-1), y : input.shape[3] - (kernel_size[1]-y-1)].flatten()
                autocorr_matrices[c,x,y] = NMIK(center_vals, neighbor_vals, k)
    return autocorr_matrices.nanmean(dim=0)

from functools import partial
import torch.nn as nn
import torch.nn.functional as F

def generate_gradient_multiplier_from_correlation(args, model, train_loader):
    model.eval()

    gradient_multipliers = {}
            
    # set correaltion measure
    if args.c_type.startswith("nmi"):
        k = float(args.c_type[3:])
        print(f"using normalized mutual information with k={k}")
        feature_map_correlation = partial(feature_map_nmik, k=k)
    else:
        raise Exception(f'correlation metric not found')

    # create forward hooks to capture intermediate feature maps
    feature_map_dict = {}
    def conv_forward_hook_factory(depth):
        def conv_forward_hook(layer, inputs, _):
            input = inputs[0]
            padded_input = F.pad(input, (layer.padding[0], layer.padding[0], layer.padding[1], layer.padding[1]), mode="constant", value=0)
            feature_map_dict[(depth, layer.kernel_size)] = padded_input.detach().cpu()
            return None
        return conv_forward_hook

    # register hooks on all convolutions with recursive model search
    hook_handles = []
    def register_conv_hooks(module, depth):
        for _, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.kernel_size[0] > 1 and child.kernel_size[1] > 1:
                conv_forward_hood = conv_forward_hook_factory(depth)
                handle = child.register_forward_hook(conv_forward_hood)
                hook_handles.append(handle)
                depth += 1
            depth = register_conv_hooks(child, depth)
        return depth
    _ = register_conv_hooks(model, depth=0)

    # get {--c_train_batches} training images
    images_list = []
    for i, (images, _) in enumerate(train_loader):   
        input = images.cuda()
        
        images_list.append(input)

        if i == (args.c_train_batches - 1):
            break

    # forward pass to activate hooks and find correlation for intermediate feature maps
    input = torch.cat(images_list)
    model(input)

    for (depth, kernel_size), feature_map in feature_map_dict.items():
        # concat feature maps of all distributed models
        if args.distributed:
            gather_feature_map = [torch.zeros_like(feature_map) for _ in range(args.world_size)]
            torch.distributed.all_gather_object(gather_feature_map, feature_map)
            feature_map = torch.cat(gather_feature_map)

            # only do correlation computation on rank == 0
            if torch.distributed.get_rank() == 0:
                corr_matrix = feature_map_correlation(feature_map, kernel_size).cuda()
                print(corr_matrix)
            else:
                corr_matrix = torch.zeros(kernel_size).cuda()
            torch.distributed.broadcast(corr_matrix, src=0)
        else:
            corr_matrix = feature_map_correlation(feature_map, kernel_size)
        
        gradient_multiplier = autocorrelation_to_gradient_multiplier(corr_matrix).cuda() # normalize correlation matrix

        if not torch.isnan(gradient_multiplier).any():
            gradient_multipliers[depth] = gradient_multiplier
        else:
            print("NAN encountered")
            print(corr_matrix, gradient_multiplier)

    # remove hooks
    for hook_handle in hook_handles:
        hook_handle.remove()
    
    return gradient_multipliers