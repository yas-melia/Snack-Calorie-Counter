<p align="center">
  <img src="Media/Illustration.png" width="600">
</p>

# 🍫 Snack-Calorie-Counter
Computer Vision project to detect Indonesian sweets using YOLOv11. Created ahead of Valentine’s Day to prevent excessive calorie intake.

# 📌 Project Overview
- Using YOLOv11 object detection
- Custom dataset to detect several sweets (7 different classes):
  - TOP
  - Beng-beng Maxx
  - Cadbury Dairy Milk
  - Delfi Chocolate
  - BAR BAR
  - SilverQueen
  - VAN HOUTEN

 # 🗂 Dataset
 The dataset consist of 7 different classes, independently collected and manually annotated using CVAT to generate bounding box labels in YOLO format.
 - Train: 961 Images
 - Val: 80 Images

# 📊 Evaluation
YOLOv11n with 100 epoch, the best model got:
|    Metric   | Score |
|-------------|-------|
| Recall      | 0.97  |
| Precission  | 0.96  |
| mAP50-95    | 0.93  |

# 📸 Detection Preview
<p align="center">
  <img src="Media/Illustration2.jpg" width="600">
</p>

# 💻 Behind This Project
This project was crafted ahead of Valentine's Day to help everyone enjoy chocolate more wisely
built with Python and powered by YOLO for real-time snack detection!
