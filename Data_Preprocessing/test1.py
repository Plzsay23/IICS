train_transform = v2.Compose([
    # Step 1: Resize (필수 유지)
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BILINEAR),

    # Step 2: 공간적 변형 강화 (RandomAffine)
    v2.RandomAffine(
        degrees=10,                      # ±10도 회전
        translate=(0.05, 0.05),          # 5% 범위 내에서 이미지 이동
        scale=(0.95, 1.05),              # 5% 범위 내에서 이미지 크기 조정
        interpolation=v2.InterpolationMode.BILINEAR,
        p=0.8
    ),

    # Step 3: 명암 노이즈 추가 (RandomAutocontrast)
    v2.RandomAutocontrast(p=0.1),
    
    # Step 4: 모델 입력용 dtype + normalize
    v2.ToDtype(torch.float32, scale=True),
    
    # Step 5: Normalize 값 변경 (MRI 데이터에 적합한 0.5/0.5 정규화)
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])