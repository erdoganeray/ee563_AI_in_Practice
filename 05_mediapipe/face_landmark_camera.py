import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import time

# MediaPipe Face Landmarker'ı başlat
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # Video modu için
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

detector = vision.FaceLandmarker.create_from_options(options)

# Kamerayı aç
cap = cv2.VideoCapture(0)

# FPS hesaplama için
frame_count = 0
start_time = time.time()
fps = 0

print("Kamera açıldı. Çıkmak için 'q' tuşuna basın.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kamera görüntüsü alınamadı.")
        break
    
    # Frame'i RGB'ye çevir (MediaPipe RGB kullanır)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Image formatına çevir
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Timestamp (milisaniye cinsinden)
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    
    # Face landmark detection
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    # Sonuçları çiz
    annotated_image = np.copy(rgb_frame)
    
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Face mesh tesselation (yüz ağı)
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Face contours (yüz konturu)
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Left iris (sol göz bebeği)
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            
            # Right iris (sağ göz bebeği)
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    
    # BGR'ye geri çevir (OpenCV BGR kullanır)
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # FPS hesaplama
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    # Her saniyede bir FPS'i güncelle
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    # FPS bilgisi ekle
    cv2.putText(annotated_image_bgr, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Yüz sayısı bilgisi
    num_faces = len(detection_result.face_landmarks) if detection_result.face_landmarks else 0
    cv2.putText(annotated_image_bgr, f'Faces: {num_faces}', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Görüntüyü göster
    cv2.imshow('MediaPipe Face Landmarks', annotated_image_bgr)
    
    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
detector.close()
print("Program sonlandırıldı.")
