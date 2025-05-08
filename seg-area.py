import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8-Seg model
model = YOLO(r"c:\Users\user\Downloads\best (1).pt")

# Directory containing test images
image_dir = r'D:\fyp dataset\dataset\test\images'
image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

total_damage_area = 0
total_image_area = 0

for idx, image_filename in enumerate(image_filenames):
    img_path = os.path.join(image_dir, image_filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image {img_path}")
        continue

    image_area = img.shape[0] * img.shape[1]
    total_image_area += image_area

    results = model(img)
    image_damage_area = 0

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()

        # Create a copy for transparent overlays
        overlay = img.copy()

        for mask, label in zip(masks, labels):
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if int(label) == 0:  # Late Blight
                color = (0, 0, 255)  # Red
                mask_area = np.sum(mask_resized)
                image_damage_area += mask_area
                total_damage_area += mask_area
            elif int(label) == 1:  # Leaf
                color = (0, 255, 0)  # Green
            else:
                continue

            # Apply transparent fill to the overlay
            for contour in contours:
                cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)

            # Draw solid boundary on the original image
            cv2.drawContours(img, contours, -1, color, thickness=3)

        # Blend original image with overlay using transparency
        alpha = 0.4  # Transparency level
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Add damage percentage text
        percentage_damage = (image_damage_area / image_area) * 100
        cv2.putText(img, f'Damage: {percentage_damage:.2f}%', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    new_image_filename = f"segmented_image_{idx + 1}.jpg"
    new_img_path = os.path.join(image_dir, new_image_filename)
    cv2.imwrite(new_img_path, img)

    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmented Damage - {new_image_filename}")
    plt.axis('off')
    plt.show()

# Final summary
total_percentage_damage = (total_damage_area / total_image_area) * 100
print(f'Total damage area: {total_damage_area} pixels²')
print(f'Total percentage damage area across all images: {total_percentage_damage:.2f}%')
