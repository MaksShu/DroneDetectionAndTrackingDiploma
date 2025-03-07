import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentations(height, width):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomSizedCrop(min_max_height=(int(height * 0.6), height),
                          height=height, width=width, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.4),
        A.Affine(shear=(-15, 15), p=0.3),

        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120)),
        A.ToGray(p=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),

        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        fill_value=0, p=0.4),
        A.Downscale(scale_min=0.5, scale_max=0.75, p=0.2),

        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.4,
        label_fields=['labels']
    ))
