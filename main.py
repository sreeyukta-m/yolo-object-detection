"""
YOLO Real-Time Object Detection
Main entry point for the application
"""

from src.yolo_detector import YOLOWebcamDetector
import argparse
import sys


def main():
    """Main function to run YOLO object detection"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Real-time object detection using YOLO'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLO model to use (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Validate confidence threshold
    if not 0.0 <= args.conf <= 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    try:
        print("=" * 60)
        print("YOLO Real-Time Object Detection")
        print("=" * 60)
        print(f"\nModel: {args.model}")
        print(f"Confidence Threshold: {args.conf}")
        print(f"Camera ID: {args.camera}")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Save current frame")
        print("  + - Increase confidence threshold")
        print("  - - Decrease confidence threshold")
        print("=" * 60)
        
        # Create detector instance
        detector = YOLOWebcamDetector(
            model_name=args.model,
            conf_threshold=args.conf
        )
        
        # Run detection
        detector.run(camera_id=args.camera)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    