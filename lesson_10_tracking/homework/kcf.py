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

tracker1 = cv2.legacy.TrackerKCF_create()
tracker2 = cv2.legacy.TrackerKCF_create()

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

# KCF (Kernelized Correlation Filters)

# 1. Основна ідея
# KCF базується на кореляційних фільтрах — це такі фільтри, які навчаються «розпізнавати» 
# цільовий об’єкт у послідовних кадрах відео шляхом пошуку зони з максимальною кореляцією (схожістю).
# На відміну від простих кореляційних фільтрів, KCF: використовує “kernel trick” 
# (ядрове перетворення), щоб розраховувати схожість у нелінійному просторі;
# застосовує швидке перетворення Фур’є (FFT) для прискорення обчислень.

# 2. Принцип роботи (спрощено)

# Навчання фільтра:
# Алгоритм будує кореляційний фільтр, який максимізує відгук (response map) у центрі об’єкта.

# Трекінг у наступних кадрах:
# Для кожного нового кадру обчислюється відгук фільтра — пік (максимум) відповідає новому положенню об’єкта.

# Оновлення фільтра:
# Модель поступово оновлюється, щоб адаптуватися до змін у зовнішньому вигляді об’єкта.

# 3. Переваги
# ✅ Дуже швидкий (до сотень кадрів/с).
# ✅ Не потребує повторного навчання.
# ✅ Простий у використанні через OpenCV.
# ✅ Добре працює при невеликих змінах масштабу чи освітлення.

# 🔹 4. Недоліки
# ❌ Погано працює при значних деформаціях або обертаннях.
# ❌ Може «загубити» об’єкт при сильному перекритті.
# ❌ Не підтримує повторне виявлення (re-detection) після втрати.
