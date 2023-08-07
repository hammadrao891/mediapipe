import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection and Face Mesh modules
mp_face_detection = mp.solutions.face_detection

# Load an input image
input_image_path = 'input_image.jpg'
input_image = cv2.imread(input_image_path)

# Initialize Face Detection and Face Mesh models
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Perform face detection
    results = face_detection.process(image_rgb)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = input_image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw bounding box around the face
            cv2.rectangle(input_image, bbox, (0, 255, 0), 2)
            
            # Initialize MediaPipe Face Mesh module
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            # Convert the image to RGB
            image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            # Perform face mesh detection
            mesh_results = face_mesh.process(image_rgb)
            
            if mesh_results.multi_face_landmarks:
                for landmarks in mesh_results.multi_face_landmarks:
                    # Get the coordinates of the landmarks for the upper and lower lips
                    upper_lip_landmarks = [61, 185, 40, 39, 37]  # Adjust indices as needed for the upper lip
                    lower_lip_landmarks = [0, 267, 270]          # Adjust indices as needed for the lower lip
                    
                    # Create polygon points for upper lip
                    upper_lip_points = [(int(landmarks.landmark[idx].x * iw), int(landmarks.landmark[idx].y * ih)) for idx in upper_lip_landmarks]
                    upper_lip_points = [upper_lip_points[0], upper_lip_points[1], upper_lip_points[2], upper_lip_points[3], upper_lip_points[4]]
                    
                    # Create polygon points for lower lip
                    lower_lip_points = [(int(landmarks.landmark[idx].x * iw), int(landmarks.landmark[idx].y * ih)) for idx in lower_lip_landmarks]
                    lower_lip_points = [lower_lip_points[0], lower_lip_points[1], lower_lip_points[2]]
                    
                    # Draw filled polygons on the lips areas
                    cv2.fillPoly(input_image, [np.array(upper_lip_points)], color=(255, 0, 0))  # Blue color for upper lip
                    cv2.fillPoly(input_image, [np.array(lower_lip_points)], color=(255, 0, 0))  # Blue color for lower lip
                    
    # Save the output image with lipstick on both lips
    output_image_path = 'output_image_with_lipstick.jpg'
    cv2.imwrite(output_image_path, input_image)

print("Output image with lipstick on both lips saved:", output_image_path)
