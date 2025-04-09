import cv2
import time

# Открытие видеофайла
cap = cv2.VideoCapture(r'C:\Users\ilyac\PycharmProjects\YOLOv8\.venv\video\person_2025-04-07_23-36-45.avi')  # Укажите путь к вашему видеофайлу

if not cap.isOpened():
    print("Ошибка при открытии видео!")
    exit()

# Получаем FPS видео (кадров в секунду)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS видео: {fps}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Если не удалось прочитать кадр, завершаем

    # Отображаем кадр
    cv2.imshow('Video', frame)

    # Добавляем задержку между кадрами для замедления видео
    # Если FPS = 30, а мы хотим воспроизводить в 2 раза медленным темпом, ставим задержку 2x
    time.sleep(1 / fps * 4)  # Умножаем на 2 для замедления воспроизведения

    # Выход из программы при нажатии клавиши 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Закрываем все окна и освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
