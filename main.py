import cv2
from ultralytics import YOLO

# Загрузим YOLOv8 модель (если ты скачал .pt файл вручную, укажи путь)
model = YOLO('yolov8n.pt')  # автоматически скачается, если файла нет

# Открываем веб-камеру
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция объектов
    results = model(frame)[0]

    for result in results.boxes:
        cls_id = int(result.cls[0])
        conf = float(result.conf[0])
        if model.names[cls_id] == 'person':
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            # Зелёная рамка
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Текст
            label = f'{model.names[cls_id]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Показываем результат
    cv2.imshow("Detection", frame)

    # Выход — по клавише ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
