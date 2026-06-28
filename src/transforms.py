import albumentations as A
from albumentations.pytorch import ToTensorV2

# WHY ALBUMENTATIONS instead of torchvision transforms? 
# faster, 
# has more medical-imaging transforms, 
#useful for segmentation


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(phase, size=380):
    if phase=="train":
        # we want the model to see variety:
        # The same lesion flipped horizontally is still the same lesion 
        # A slightly rotated lesion is still the same lesion
        #A lesion under slightly different lighting is still the same lesion
        return A.Compose([A.Resize(size, size), 
                          A.HorizontalFlip(p=0.5),
                          A.VerticalFlip(p=0.5),
                          A.Transpose(p=0.5), 
                          A.ShiftScaleRotate(
                                shift_limit=0.1,
                                scale_limit=0.1,
                                rotate_limit=45,
                                p=0.5
                            ),
                          A.HueSaturationValue(
                                hue_shift_limit=10,
                                sat_shift_limit=20,
                                val_shift_limit=10,
                                p=0.5
                            ),
                            A.RandomBrightnessContrast(
                                brightness_limit=0.2,
                                contrast_limit=0.2,
                                p=0.5
                            ),
                            A.CoarseDropout(
                                max_holes=8,
                                max_height=size // 16,
                                max_width=size // 16,
                                fill_value=0,
                                p=0.5
                            ),
                            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                            ToTensorV2(),
                            ])
    else:
         # Validation pipeline: only the essential steps
        return A.Compose([A.Resize(size,size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])



def get_tta_transforms(size: int = 380):
    #Test-Time Augmentation (TTA) pipeline
    #we predict multiple times with slight random augmentations and average the results
    # The model might miss something from one angle but catch it from another
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])