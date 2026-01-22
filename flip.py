import os
import random
from PIL import Image
from torchvision import transforms

# ================= CONFIG =================
INPUT_DIR = "hq_train"
OUTPUT_DIR = "hq_train_augmented"
TARGET_IMAGES_PER_CLASS = 6000
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================

# Augmentations: SAFE (NO degradation)
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.9, 1.0),
        ratio=(0.95, 1.05)
    ),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05,
        hue=0.02
    )
])

def is_image(file):
    return file.lower().endswith(IMG_EXTENSIONS)

for class_name in os.listdir(INPUT_DIR):
    class_input_path = os.path.join(INPUT_DIR, class_name)
    class_output_path = os.path.join(OUTPUT_DIR, class_name)

    if not os.path.isdir(class_input_path):
        continue

    os.makedirs(class_output_path, exist_ok=True)

    images = [f for f in os.listdir(class_input_path) if is_image(f)]
    original_count = len(images)

    if original_count == 0:
        continue

    # 1️⃣ Copy original HQ images
    for img_name in images:
        img = Image.open(os.path.join(class_input_path, img_name)).convert("RGB")
        img.save(os.path.join(class_output_path, img_name))

    print(f"[{class_name}] Copied {original_count} HQ images")

    # 2️⃣ Augment to reach target count
    needed = TARGET_IMAGES_PER_CLASS - original_count
    idx = 0

    while needed > 0:
        img_name = random.choice(images)
        img_path = os.path.join(class_input_path, img_name)

        img = Image.open(img_path).convert("RGB")
        aug_img = augmentation(img)

        new_name = f"aug_{idx}_{img_name}"
        aug_img.save(os.path.join(class_output_path, new_name))

        idx += 1
        needed -= 1

    print(f"[{class_name}] Augmented to {TARGET_IMAGES_PER_CLASS} images")

print("✅ HQ-only augmentation completed cleanly.")
