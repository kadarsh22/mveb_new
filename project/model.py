import copy
import math 
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(cfg):
    model = SiameseNet(cfg)
    if torch.distributed.is_available():
        # SyncBN optional: ebreg runs with --no-sync-batchnorm (per-GPU BN stats),
        # so set sync_batchnorm: False in the config to match it
        if getattr(cfg, 'sync_batchnorm', True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        # channels_last is the layout cuDNN conv kernels want for a ResNet under
        # bf16 autocast (matches ebreg); inputs are converted in the train loop.
        if getattr(cfg, 'channels_last', False):
            model = model.to(memory_format=torch.channels_last)
        # Module.compile() compiles in place, so state_dict keys stay
        # 'module.encoder.*' (torch.compile(model) would add '_orig_mod.' and
        # break resume / lineval.py loading).
        if getattr(cfg, 'compile', False):
            model.compile()
        
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[torch.cuda.current_device()],
                                                          # EMA branch params never get grads;
                                                          # shared encoder uses every param
                                                          find_unused_parameters=not getattr(cfg, 'shared_encoder', False))
    else:
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    return model


class SiameseNet(nn.Module):
    """
    Build a Siamese model.
    Some codes are borrowed from MoCo & SimSiam.
    """

    def __init__(self, cfg):
        super(SiameseNet, self).__init__()
        self.cfg = cfg
        
        zero_init_residual = getattr(self.cfg, 'zero_init_residual', True)
        net = getattr(models.resnet, cfg.arch)(zero_init_residual=zero_init_residual)
        prev_dim = net.fc.weight.shape[1]
        # build online branch
        self.encoder = nn.Sequential(*list(net.children())[:-1] + [nn.Flatten(1)])
        self.projector = _projection_mlp(prev_dim,
                                            self.cfg.projector_hidden_dim,
                                            self.cfg.projector_output_dim)

        # shared_encoder=True: VICReg-style, both views go through the same
        # encoder/projector with gradients and no momentum (EMA) target branch.
        self.shared_encoder = getattr(self.cfg, 'shared_encoder', False)
        if not self.shared_encoder:
            # build target branch
            self.momentum_encoder = copy.deepcopy(self.encoder)
            self.momentum_projector = copy.deepcopy(self.projector)
        
    # kept out of the compiled graph: mm is a Python float that changes every
    # step, and dynamo would otherwise recompile on each new value
    @torch.compiler.disable
    @torch.no_grad()
    def _momentum_update_key_encoder(self, mm):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)

    def forward(self, image, mm):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https:/ /arxiv.org/abs/2011.10566 for detailed notations
        """  
        # online branch
        z_f = []
        # detached backbone features of every view (no extra compute): feats[0]
        # feeds the online probe (as ebreg's emb_x.detach()), and the first two
        # feed the online effective rank (as ebreg's cat(emb_x, emb_y))
        feats = []

        for image_i in image:
            y = self.encoder(image_i)
            feats.append(y.detach())
            z = self.projector(y)
            z=torch.nn.functional.normalize(z)
            z_f.append(z)

        if self.shared_encoder:
            # targets are the online embeddings of the two global views
            # (not detached, so gradients flow through both branches)
            return z_f, z_f[0], z_f[1], feats

        # target branch
        with torch.no_grad():
            self._momentum_update_key_encoder(mm)
            y1m = self.momentum_encoder(image[0])
            y2m = self.momentum_encoder(image[1])
            z1m = self.momentum_projector(y1m)
            z2m = self.momentum_projector(y2m)
        return z_f, torch.nn.functional.normalize(z1m), torch.nn.functional.normalize(z2m), feats


def _projection_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int,
                    bias: bool = False) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with
    BN applied to its hidden fc layers but no ReLU on the output fc layer.
    The CIFAR-10 study used a MLP with only two layers.

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims:
            Hidden dimension of all the fully connected layers.
        out_dims:
            Output Dimension of the final linear layer.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims, bias=bias),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims, bias=bias),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims))

    projection = nn.Sequential(l1, l2, l3)
    
    return projection