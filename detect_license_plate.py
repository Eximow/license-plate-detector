import cv2
from ultralytics import YOLO

def detect_and_show_plate(image_path, model_path="best.pt", output_path="plate_output.jpg"):
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(img)[0]

    plate_roi = None
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        plate_roi = img[y1:y2, x1:x2]

        # Draw a thick red rectangle around the plate
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)

    if plate_roi is not None and plate_roi.size > 0:
        # Enlarge the plate
        plate_big = cv2.resize(plate_roi, (300, 100), interpolation=cv2.INTER_CUBIC)

        # Place the enlarged plate at the top-right corner of the image
        h, w, _ = img.shape
        ph, pw, _ = plate_big.shape

        # Coordinates for top-right placement
        x_offset = w - pw - 10   # 10 pixels from the right
        y_offset = 10            # 10 pixels from the top

        # Paste the enlarged plate inside the image
        img[y_offset:y_offset+ph, x_offset:x_offset+pw] = plate_big

    cv2.imwrite(output_path, img)
    print(f"✅ Output saved: {output_path}")

if __name__ == "__main__":
    detect_and_show_plate("car.jpg", model_path="best.pt", output_path="plate_output.jpg")
