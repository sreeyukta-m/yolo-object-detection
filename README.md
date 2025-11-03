# YOLO Real-Time Object Detection 🚀

A real-time object detection system using YOLOv8 for webcam feed analysis. This project was developed as part of the Computer Vision and Image Processing (CVIP) course.

## 📋 Features

- ✅ Real-time object detection from webcam
- ✅ Object counting and tracking
- ✅ FPS monitoring
- ✅ Adjustable confidence threshold
- ✅ Save detection results
- ✅ Multi-class detection (80 COCO classes)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/sreeyukta-m/yolo-object-detection.git
cd yolo-object-detection
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage
```bash
python main.py
```

### Keyboard Controls
- **Q** - Quit
- **S** - Save current frame
- **+** - Increase confidence threshold
- **-** - Decrease confidence threshold

## 📊 Performance

| Model | Size | Speed (FPS)* |
|-------|------|--------------|
| YOLOv8n | 6 MB | ~45 |
| YOLOv8s | 22 MB | ~35 |
| YOLOv8m | 50 MB | ~25 |

## 📧 Contact

 sreeyukta-m - 23btrad010@jainuniversity.ac.in


Project Link: [https://github.com/sreeyukta/yolo-object-detection](https://github.com/YOUR_USERNAME/yolo-object-detection)