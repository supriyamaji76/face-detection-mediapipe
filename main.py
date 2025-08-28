import os
import argparse
import cv2
import mediapipe as mp

# Mediapipe face detection model is made by Google. It uses a lightweight model for real-time face detection.
# Reference: https://google.github.io/mediapipe/solutions/face_detection.html

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Ensure the crop doesn't go out of bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x1 + w), min(H, y1 + h)

            # Blur the detected face region
            img[y1:y2, x1:x2] = cv2.blur(img[y1:y2, x1:x2], (30, 30))

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="webcam", choices=["webcam", "image", "video"]
    )
    parser.add_argument("--filePath", default=None, help="Path to image or video file")
    args = parser.parse_args()

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.3
    ) as face_detection:
        if args.mode == "image":
            if args.filePath is None:
                print("❌ Error: Image path not provided.")
                return
            img = cv2.imread(args.filePath)
            if img is None:
                print("❌ Error: Could not read the image.")
                return

            processed_img = process_img(img, face_detection)
            out_path = os.path.join(output_dir, "output.png")
            cv2.imwrite(out_path, processed_img)
            print(f"✅ Processed image saved at {out_path}")

        elif args.mode == "video":
            if args.filePath is None:
                print("❌ Error: Video path not provided.")
                return
            cap = cv2.VideoCapture(args.filePath)
            if not cap.isOpened():
                print("❌ Error: Could not open video file.")
                return

            ret, frame = cap.read()
            H, W = frame.shape[:2]
            out_path = os.path.join(output_dir, "output.mp4")
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"avc1"), 25, (W, H)
            )

            while ret:
                frame = process_img(frame, face_detection)
                writer.write(frame)
                ret, frame = cap.read()

            cap.release()
            writer.release()
            print(f"✅ Processed video saved at {out_path}")

        elif args.mode == "webcam":
            cap = cv2.VideoCapture(0)  # Use 0 for MacBook's internal webcam
            if not cap.isOpened():
                print("❌ Error: Could not access webcam.")
                return

            print("📷 Press 'q' to quit webcam window.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = process_img(frame, face_detection)
                cv2.imshow("Webcam - Face Blur", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
