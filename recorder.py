import cv2
import time
import os
from ultralytics import YOLO
import logging

# Отключаем вывод ненужных логов из ultralytics
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Загружаем модель YOLOv8
model = YOLO('yolov8n.pt')

# Захват видео с камеры
cap = cv2.VideoCapture(0)

# Устанавливаем разрешение для захвата кадров
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Получаем FPS из камеры
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS камеры: {fps}")

# Установим FPS записи видео равным FPS камеры
record_fps = fps  # Записываем с FPS, который камера реально поддерживает

# Время между кадрами (в секундах)
frame_interval = 1 / record_fps

recording = False
record_start_time = 0
out = None
last_frame_time = 0

# Путь для сохранения видео
save_dir = r'C:\Users\ilyac\PycharmProjects\YOLOv8\.venv\video'
os.makedirs(save_dir, exist_ok=True)  # Создаст папку, если не существует

# Используем кодек MJPG для записи видео с хорошим качеством
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Применяем модель YOLOv8 для обнаружения объектов
    results = model(frame)[0]
    person_detected = False

    # Проходим по результатам и рисуем рамки вокруг людей
    for result in results.boxes:
        cls_id = int(result.cls[0])
        if model.names[cls_id] == 'person':
            person_detected = True
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            # Уменьшаем ширину рамки на 20%
            width_shrink = int((x2 - x1) * 0.2)
            x1 += width_shrink
            x2 -= width_shrink
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Начинаем запись, если человек был обнаружен
    if person_detected and not recording:
        print("🟢 Обнаружен человек — начинаю запись на 10 секунд")
        record_start_time = time.time()
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        video_path = os.path.join(save_dir, f'person_{timestamp}.avi')
        out = cv2.VideoWriter(video_path, fourcc, record_fps, (frame.shape[1], frame.shape[0]))
        recording = True

    # Запись видео с интервалом времени
    if recording:
        current_time = time.time()

        # Пишем кадр в видео, если прошло достаточно времени
        if current_time - last_frame_time >= frame_interval:
            out.write(frame)
            last_frame_time = current_time

        # Проверяем, прошло ли 10 секунд
        if current_time - record_start_time >= 10:
            print("🔴 10 секунд прошло — запись завершена")
            recording = False
            out.release()
            out = None

    # Отображаем видео в окне
    cv2.imshow("Live", frame)

    # Выход по клавише ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Завершаем работу с камерой и записывающим устройством
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
