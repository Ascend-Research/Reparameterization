import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fast_histogram import histogram2d

def dependence_to_spatial_gradient_scaling(corr_matrix):
    spatial_gradient_scaling = corr_matrix.clone()
    spatial_gradient_scaling /= torch.mean(spatial_gradient_scaling, dim=(-2,-1), keepdim=True)
    return spatial_gradient_scaling

def shannon_entropy(c):
        c = c[np.nonzero(c)]
        c /= np.sum(c)
        H = -np.sum(c * np.log2(c))  
        return H
   
def NMIK(X, Y, k, diag_rem):
    bins = int(1 + np.ceil(np.log2(len(X))))    
    
    range_X = [np.min(X), np.max(X) + 1e-12]
    range_Y = [np.min(Y), np.max(Y) + 1e-12]
    
    c_XY = histogram2d(X, Y, range=[range_X, range_Y], bins=bins)
    if diag_rem:
        c_XY *= np.ones((bins, bins)) - np.eye(bins)
    c_X = np.sum(c_XY, axis=0)
    c_Y = np.sum(c_XY, axis=1)

    H_X = shannon_entropy(c_X)
    H_Y = shannon_entropy(c_Y)
    H_XY = shannon_entropy(c_XY.flatten())

    MI = H_X + H_Y - H_XY
    NMI = MI/H_XY
    return k * NMI / ((k-1)*NMI + 1)

def feature_map_nmik(input, kernel_size=(3,3), k=5, diag_rem=False):
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

    input = input.numpy()
    _, channels, height, width = input.shape

    autocorr_matrices = torch.empty(channels, *kernel_size)
    autocorr_matrices[:, int((kernel_size[0] - 1)/2), int((kernel_size[1] - 1)/2)] = 1 # set self center to 1

    all_ds = []
    for d_x in range(-int((kernel_size[0] - 1)/2), int((kernel_size[0] - 1)/2) + 1):
        for d_y in range(-int((kernel_size[1] - 1)/2), int((kernel_size[1] - 1)/2) + 1):
            all_ds.append((d_x, d_y))
    ds = []
    for d_x, d_y in all_ds:
        if (-d_x, -d_y) not in ds and (d_x, d_y) != (0,0): 
            ds.append((d_x, d_y))

    for c in range(channels):
        input_c = input[:, c, :, :]

        for (d_x, d_y) in ds:
            x_0 = int((kernel_size[0] - 1)/2) + d_x
            y_0 = int((kernel_size[0] - 1)/2) + d_y

            c_x = -d_x * (d_x < 0)
            c_y = -d_y * (d_y < 0)

            n_x = c_x + d_x
            n_y = c_y + d_y

            center_vals = input_c[:,c_x:height-n_x, c_y:width-n_y].flatten()
            neighbor_vals = input_c[:,n_x:height-c_x, n_y:width-c_y].flatten()

            if not diag_rem:
                mi = NMIK(center_vals, neighbor_vals, k, diag_rem)
                autocorr_matrices[c,x_0,y_0] = mi
            else:
                # get better bin statistics by removing repeated zeros
                non_zero_loc = (center_vals != 0) * (neighbor_vals != 0)
                if np.sum(non_zero_loc) != 0:
                    mi = NMIK(center_vals[non_zero_loc], neighbor_vals[non_zero_loc], k, diag_rem)
                    if not np.isnan(mi):
                        autocorr_matrices[c,x_0,y_0] = mi
                    else:
                        autocorr_matrices[c,x_0,y_0] = 0
                else:
                    autocorr_matrices[c,x_0,y_0] = 0

            x_1 = int((kernel_size[0] - 1)/2) - d_x
            y_1 = int((kernel_size[0] - 1)/2) - d_y
            autocorr_matrices[c,x_1,y_1] = autocorr_matrices[c,x_0,y_0]

    return autocorr_matrices.nanmean(dim=0)

def generate_spatial_gradient_scaling_from_dependence(args, model, train_loader):
    model.eval()

    spatial_gradient_scalings = {}
    
    print(f"using normalized mutual information with k={args.sgs_k}")

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

    # get {--sgs_train_batches} training images
    images_list = []
    for i, (images, _) in enumerate(train_loader):   
        input = images.cuda()
        
        images_list.append(input)

        if i == (args.sgs_train_batches - 1):
            break

    # forward pass to activate hooks and find dependence for intermediate feature maps
    input = torch.cat(images_list)
    model(input)

    for (depth, kernel_size), feature_map in feature_map_dict.items():
        # concat feature maps of all distributed models
        if args.distributed:
            gather_feature_map = [torch.zeros_like(feature_map) for _ in range(args.world_size)]
            torch.distributed.all_gather_object(gather_feature_map, feature_map)
            feature_map = torch.cat(gather_feature_map)

            # only do dependence computation on rank 0
            if torch.distributed.get_rank() == 0:
                corr_matrix = feature_map_nmik(feature_map, kernel_size, k=args.sgs_k, diag_rem=args.sgs_diagonal_removal).cuda()
            else:
                corr_matrix = torch.zeros(kernel_size).cuda()
            torch.distributed.broadcast(corr_matrix, src=0)
        else:
            corr_matrix = feature_map_nmik(feature_map, kernel_size, k=args.sgs_k, diag_rem=args.sgs_diagonal_removal)
        
        spatial_gradient_scaling = dependence_to_spatial_gradient_scaling(corr_matrix).cuda() # normalize dependence matrix

        if not torch.isnan(spatial_gradient_scaling).any():
            spatial_gradient_scalings[depth] = spatial_gradient_scaling
        else:
            print("NAN encountered")
            print(corr_matrix, spatial_gradient_scaling)

    # remove hooks
    for hook_handle in hook_handles:
        hook_handle.remove()
    
    return spatial_gradient_scalings