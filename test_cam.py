import cv2
import time

cap = cv2.VideoCapture(0)
frames = 0
start_time = time.time()

print("⏱️ Считаем кадры в течение 3 секунд...")

while time.time() - start_time < 3:
    ret, frame = cap.read()
    if not ret:
        break
    frames += 1

cap.release()
elapsed = time.time() - start_time
fps = frames / elapsed
print(f"📸 Реальный FPS камеры: {fps:.2f}")
