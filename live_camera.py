import cv2 #مكتبه نستخمها للأشياء المتعلقه بالكاميرا
import numpy as np # مكتبه نستخدمها للتعامل مع الارقام والمصفوفات
import tensorflow as tf #مكتبه نستخدمها لتحميل المودل المدرب ثم نستعمل المودل للتوقع 
import os #نستخدمها للتأكد من ان ملف المودل موجود
from collections import deque #مكتبة نستخدمها لجعل التنبؤ اكثر انسيابيه

# =========================
# Load Trained Model
# =========================

MODEL_PATH = "rock_paper_scissors_model.keras" #مسار المودل المدرب


if not os.path.exists(MODEL_PATH): #للتأكد من ان المودل موجود
    print("Model file not found.")
    print("Make sure rock_paper_scissors_model.keras exists in the same folder.")
    exit() # إذا لم يكن المودل موجودًا، يوقف البرنامج لأننا لا نستطيع عمل prediction بدون المودل
model = tf.keras.models.load_model(MODEL_PATH) #تحميل المودل الي تم تدريبه 

class_names = ["rock", "paper", "scissors"] #المودل لما يتوقع ما يرجع النتيجة على طول هو يعطيني احتمالات لكل اندكس الاندكس الاعلى هو المتوقع

IMG_SIZE = 128 #حجم الصوره نفس الحجم الي بالتدريب
prediction_history = deque(maxlen=10) # التحسين الي يعرض الناتج الاكثر تكرار
print("Model loaded successfully")
print("Press Q to quit")


# =========================
# Open Camera
# =========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # فتح الكاميرا الأساسية باستخدام DirectShow backend المناسب لويندوز
if not cap.isOpened(): #للتأكد من ان الكاميرا شغاله
    print("Could not open camera")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    # Flip frame عشان يكون طبيعي مثل المراية
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # نحدد مربع في النص عشان تحط يدك داخله
    box_size = 300
    x1 = width // 2 - box_size // 2
    y1 = height // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # قص منطقة اليد فقط
    roi = frame[y1:y2, x1:x2]

    # OpenCV يقرأ الصور BGR، لكن TensorFlow غالبًا يتعامل مع RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # تجهيز الصورة مثل التدريب
    img = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    predictions = model.predict(img, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    if confidence >= 0.70:
      prediction_history.append(class_names[predicted_index])
    else:
      prediction_history.append("Uncertain")

    predicted_class = max(set(prediction_history), key=prediction_history.count)

    # رسم المربع
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # كتابة النتيجة
    text = f"{predicted_class} ({confidence * 100:.2f}%)"

    cv2.putText(
        frame,
        text,
        (x1, y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Put your hand inside the box",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Rock Paper Scissors Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()