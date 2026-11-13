import random

from PIL import ImageFilter
from PIL import Image, ImageOps
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops



class DataAugmentationVICReg(object):
    """Two-view augmentation from VICReg
    (https://github.com/facebookresearch/vicreg/blob/main/augmentations.py)."""
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image):
        return [self.transform(image), self.transform_prime(image)]


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )




def build_dataloader(cfg):
    debug = getattr(cfg, 'debug', False)

    train_transform = DataAugmentationVICReg()

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root=cfg.train_datadir, transform=train_transform)
    memory_data = datasets.ImageFolder(root=cfg.train_datadir, transform=test_transform)
    test_data = datasets.ImageFolder(root=cfg.test_datadir, transform=test_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    memory_sampler = torch.utils.data.distributed.DistributedSampler(memory_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    batch_size = int(cfg.whole_batch_size / torch.distributed.get_world_size())
   
    
    train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.n_workers,
            sampler=train_sampler,
            drop_last=True,
            persistent_workers=True,
            # 1, as ebreg main_vicreg.py (its OOM workaround); the default 2 would
            # let MVEB hide more loader stalls and skew the time comparison
            prefetch_factor=1,
        )
    memory_loader = torch.utils.data.DataLoader(
            dataset=memory_data,
            batch_size=512,
            shuffle=False,
            num_workers=cfg.n_workers,
            sampler=memory_sampler,
        )
    test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=256,
            shuffle=False,
            num_workers=cfg.n_workers,
            sampler=test_sampler
        )
    
    return train_loader, memory_loader, test_loader