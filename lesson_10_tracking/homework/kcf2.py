import cv2

cap = cv2.VideoCapture('../data/video2.mp4')
ret, frame = cap.read()

#bbox = (x, y, w, h)
x1, y1 = 232, 145
x2, y2 = 287, 190
bbox1 = (232, 145, 55, 45)

x3, y3 = 343, 157
x4, y4 = 404, 203
bbox2 = (343, 157, 61, 46)

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
