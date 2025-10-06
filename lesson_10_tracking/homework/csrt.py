import cv2

cap = cv2.VideoCapture('../data/slow_traffic.mp4')
ret, frame = cap.read()

#bbox = (x, y, w, h)
x1, y1 = 222, 121
x2, y2 = 278, 169
bbox1 = (222, 121, 56, 48)

x3, y3 = 259, 73
x4, y4 = 308, 121
bbox2 = (259, 73, 49, 48)

tracker1 = cv2.legacy.TrackerCSRT_create()
tracker2 = cv2.legacy.TrackerCSRT_create()

tracker1.init(frame, bbox1)
tracker2.init(frame, bbox2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success1, box1 = tracker1.update(frame)
    success2, box2 = tracker2.update(frame)

    if success1:
        x, y, w, h = [int(v) for v in box1]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    if success2:
        x2, y2, w2, h2 = [int(v) for v in box2]
        cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (255,0,0), 2)
    else:
        cv2.putText(frame, "Lost", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

    cv2.imshow("Multi-Object KCF Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#CSRT (Channel and Spatial Reliability Tracker)
#альтернатива KCF, але більш точний.

# Основна ідея

# CSRT покращує KCF за рахунок використання просторової та канальної надійності:
# Канальна надійність (Channel Reliability)
# Об’єкт представлений у різних каналах (наприклад, колір, градієнт).
# Кожен канал має свою "вагу" залежно від того, наскільки він корисний для трекінгу.
# Просторова надійність (Spatial Reliability)
# Алгоритм визначає, які пікселі всередині ROI найбільш надійні для трекінгу, 
# а які можуть заважати (шум, фон).
# Поєднання з кореляційними фільтрами
# CSRT використовує кореляційні фільтри як KCF, але з додатковою просторовою 
# маскою для кращої точності.

# Переваги CSRT
# ✅ Вища точність, особливо при змінах масштабу та часткових перекриттях.
# ✅ Працює з деформацією об’єкта краще, ніж KCF.
# ✅ Підтримує адаптивне оновлення ROI.

# 🔹 3. Недоліки
# ❌ Повільніший, ніж KCF (не завжди підходить для відео в реальному часі на слабкому CPU).
# ❌ Більш складний, більше ресурсів на обчислення.