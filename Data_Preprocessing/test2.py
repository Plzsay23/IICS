train_transform = v2.Compose([
    # v2.ToImage(),  # 제거 가능
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BILINEAR),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=3),  # 5 -> 3
    v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    v2.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])