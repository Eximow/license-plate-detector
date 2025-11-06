
# License Plate Detector 🚗🔍

This project is a Python application that uses **YOLO (You Only Look Once)** and **OpenCV** to detect car license plates in images.  
The program draws a **thick red bounding box** around detected plates and also shows an **enlarged version of the plate** in the top-right corner of the image.

---

input :
![car](https://github.com/user-attachments/assets/33702e3a-cd3c-48f8-a8dd-f3a78a2e222b)


output :
![plate_output](https://github.com/user-attachments/assets/555db695-f551-4146-b761-d368430ced2c)



## ⚙️ Requirements
- Python 3.8+
- OpenCV
- ultralytics (YOLOv8)
- A trained YOLO model file (`best.pt`)

## 📂 What is `best.pt`?
The file **`best.pt`** is the trained YOLO model weights.  
It contains all the learned parameters from training on a dataset of car license plates.  
Without this file, the program cannot detect plates.  

You can either:
- Train your own YOLO model on a license plate dataset, or
- Use a pre-trained `best.pt` file that already knows how to detect plates.

---

## 🚀 Usage
1. Place your `best.pt` file in the project folder.
2. Put an image of a car (e.g., `car.jpg`) in the same folder.
3. Run the script:

