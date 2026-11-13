seed: 5 # random set
launcher: pytorch # pytorch or slurm
port: 28500 # distributed port

# dataset
train_datadir: '/leonardo_scratch/fast/EUHPC_D36_070/datasets/imagenet1k/imagenet/train'
test_datadir: '/leonardo_scratch/fast/EUHPC_D36_070/datasets/imagenet1k/imagenet/val'
n_workers: 9  # per GPU, matches ebreg ciyr9vkb (10 CPUs per task)

# model
arch: 'resnet50'
projector_hidden_dim: 8192
projector_output_dim: 256
num_layers: 3
base_momentum: 0.996
shared_encoder: True  # VICReg-style shared encoder, no EMA target (base_momentum unused)
sync_batchnorm: False  # per-GPU BN stats, matches ebreg --no-sync-batchnorm
channels_last: True  # matches ebreg
compile: True  # torch.compile, matches ebreg --compile
# resume_path: 

# optimizer
base_lr: 0.3
whole_batch_size: 512
momentum: 0.9
weight_decay: 1.0e-6
epochs: 100
warmup_epochs: 10

# loss
loss: 'loss_mveb'
lambd: 0.01  # balance factor for neg grad


# online linear probe (same as ebreg ciyr9vkb: --probe-layernorm --probe-lr 0.01 --probe-val-every 5)
probe_lr: 0.01
probe_layernorm: True
probe_val_every: 5

# others
print_freq: 100  # also the probe train-acc window; ebreg log_freq 100
test_freq: 10
save_freq: 10

# knn config
knn_k: 200
knn_t: 0.1
knn_eval: False