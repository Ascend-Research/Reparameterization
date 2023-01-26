Baseline code is taken from DyRep paper (https://github.com/hunto/image_classification_sota) with added option for gradient multiplier (--gram)

Summary:
The gradient multiplier is a KxK matrix (for KxK convolution) that is elementwise multiplied to convolution gradients to emphasize certain kernel elements. We determine the gradient multiplier for a given convolution layer by examining the correlation (mutual information) of the input feature map. For every channel in the input feature map, we find the mutual information between a pixel and its neighbors (we do this across batches by looking at neighbor pairs across all the samples in a batch). If the convolution kernel is a 3x3 we care only about its one-hop neighbor (8 neighbors (top, left, right, bottom, top left, ...)), while for larger kernels, for example, 5x5 we care about its 2 hop neighbors (24 neighbors). We order these correlation values into a KxK correlation matrix where the center element is the correlation of every center pixel with itself, and the left element is the correlation of every center pixel to its left pixel. For a correlation metric, we use normalized mutual information k which is between 0 (no correlation) and 1 (perfect correlation) with a hyperparameter k which scales intermediate values (higher k means lower correlation shows up as a higher value) (note: mutual information between a pixel and itself is 1). We then average across all channels to get a single KxK correlation matrix for a convolution layer. We can get the gradient multiplier matrix from the correlation matrix by normalizing it so the average lr is 1 (this is multiplying all elements by a constant).

args:
parser.add_argument('--gram', action='store_true', default=False, help='Use Correlation Gradient Multiplier')
parser.add_argument('--c_type', default="nmi8") # correlation measure (use nmi# (normalized mutual information #) (can be decimal)) 
parser.add_argument('--c_epochs', default=1, type=int) # update correlation and gradient multiplier every n epochs
parser.add_argument('--c_train_batches', default=2, type=int) # update correlation from n training batches (note we use 2 to simplify computational complexity)
parser.add_argument('--c_warmup', default=1, type=int) # allow n epochs of warmup training before using gradient multiplier

Algorithm Summary:
1. Every {--c_epochs} epochs calculate new gradient multipliers for each convolution from their input feature maps. Note: {--c_warmup} allows us to delay the gradient multiplier to allow the model to train a bit from random initialization.
    1.  Randomly get {--c_train_batches} number of batches from the training set and forward pass to get intermediate feature maps.
    2.  for conv in model.convolutions:
        1.  for c in conv.input.channels:
            1. Calculate {--c_type} correlation matrix of the training set for channel c (Correlation(conv.input[c,:,:]))
        2.  Average correlation matrix over all channels.

Format:
bash tools/dist_train.sh {num gpus} {hyperparameter file} {model} --experiment {folder name} --data-path {dataset path} --dist-port {open port} --gram --c_type nmi{k hyperparameter} --c_train_batches {number of training batch to compute correlation}

Output log is under experiment/{folder name}/log...



Experiments To Run:

models = [resnet18, resnet34, resnet50, MobileNetV1]
k value range: k = [1,2,4,8,16,32,64,256]

ResNet-18 ImageNet224 K Study (approx 8 min per epoch)
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet18 --experiment imagenet_res18_k1 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi1 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet18 --experiment imagenet_res18_k2 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi2 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet18 --experiment imagenet_res18_k4 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi4 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet18 --experiment imagenet_res18_k8 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi8 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet18 --experiment imagenet_res18_k16 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi16 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet18 --experiment imagenet_res18_k32 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi32 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet18 --experiment imagenet_res18_k64 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi64 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet18 --experiment imagenet_res18_k256 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi256 --c_train_batches 2


For larger resnets maybe start K search from optimal k of resnet-18


ResNet-34 ImageNet224 K Study (approx 8 min per epoch)
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet34 --experiment imagenet_res34_k1 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi1 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet34 --experiment imagenet_res34_k2 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi2 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet34 --experiment imagenet_res34_k4 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi4 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet34 --experiment imagenet_res34_k8 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi8 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet34 --experiment imagenet_res34_k16 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi16 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet34 --experiment imagenet_res34_k32 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi32 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet34 --experiment imagenet_res34_k64 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi64 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet34 --experiment imagenet_res34_k256 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi256 --c_train_batches 2


ResNet-50 ImageNet224 K Study (approx 10 min per epoch)
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet50 --experiment imagenet_res50_k1 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi1 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet50 --experiment imagenet_res50_k2 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi2 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet50 --experiment imagenet_res50_k4 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi4 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet50 --experiment imagenet_res50_k8 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi8 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet50 --experiment imagenet_res50_k16 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi16 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet50 --experiment imagenet_res50_k32 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi32 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet50 --experiment imagenet_res50_k64 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi64 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/resnet.yaml resnet50 --experiment imagenet_res50_k256 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi256 --c_train_batches 2


MobileNet ImageNet224 K Study (approx 6 min per epoch)
bash tools/dist_train.sh 8 configs/strategies/GraM/mbv1.yaml mobilenet_v1 --experiment imagenet_mbv1_k1 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi1 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/mbv1.yaml mobilenet_v1 --experiment imagenet_mbv1_k2 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi2 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/mbv1.yaml mobilenet_v1 --experiment imagenet_mbv1_k4 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi4 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/mbv1.yaml mobilenet_v1 --experiment imagenet_mbv1_k8 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi8 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/mbv1.yaml mobilenet_v1 --experiment imagenet_mbv1_k16 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi16 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/mbv1.yaml mobilenet_v1 --experiment imagenet_mbv1_k32 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi32 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/mbv1.yaml mobilenet_v1 --experiment imagenet_mbv1_k64 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi64 --c_train_batches 2
bash tools/dist_train.sh 8 configs/strategies/GraM/mbv1.yaml mobilenet_v1 --experiment imagenet_mbv1_k256 --data-path ~/datasets/ImageNet --dist-port 29500 --gram --c_type nmi256 --c_train_batches 2
