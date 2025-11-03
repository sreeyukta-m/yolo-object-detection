import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict

class YOLOWebcamDetector:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize YOLO webcam detector
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            conf_threshold: Confidence threshold for detections
        """
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.object_counts = defaultdict(int)
        
        # Color palette for different classes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
        
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        self.object_counts.clear()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf < self.conf_threshold:
                    continue
                
                # Get class name and color
                class_name = self.model.names[cls]
                color = tuple(map(int, self.colors[cls]))
                
                # Count objects
                self.object_counts[class_name] += 1
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f'{class_name} {conf:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_height - 10), 
                    (x1 + label_width, y1), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
        
        return frame
    
    def draw_info_panel(self, frame, fps, frame_count):
        """Draw information panel with FPS and object counts"""
        panel_height = 150 + len(self.object_counts) * 25
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        cv2.putText(
            frame, 
            f'FPS: {fps:.1f}', 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Frame count
        cv2.putText(
            frame, 
            f'Frames: {frame_count}', 
            (20, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        # Object counts
        cv2.putText(
            frame, 
            'Detected Objects:', 
            (20, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 0), 
            2
        )
        
        y_offset = 130
        for obj_class, count in sorted(self.object_counts.items()):
            cv2.putText(
                frame, 
                f'{obj_class}: {count}', 
                (30, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            y_offset += 25
        
        return frame
    
    def run(self, camera_id=0, window_name='YOLO Object Detection'):
        """
        Run real-time object detection
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            window_name: Name of the display window
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nStarting detection...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press '+' to increase confidence threshold")
        print("Press '-' to decrease confidence threshold")
        
        frame_count = 0
        fps = 0
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Draw detections
            frame = self.draw_detections(frame, results)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Draw info panel
            frame = self.draw_info_panel(frame, fps, frame_count)
            
            # Display confidence threshold
            cv2.putText(
                frame, 
                f'Conf: {self.conf_threshold:.2f}', 
                (frame.shape[1] - 150, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 255), 
                2
            )
            
            # Show frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'results/detection_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif key == ord('+') or key == ord('='):
                self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                print(f"Confidence threshold: {self.conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                print(f"Confidence threshold: {self.conf_threshold:.2f}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames processed: {frame_count}")


if __name__ == "__main__":
    detector = YOLOWebcamDetector(
        model_name='yolov8n.pt',
        conf_threshold=0.5
    )
    detector.run(camera_id=0)