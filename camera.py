import cv2
import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

label = {
    'men': 0,
    'women': 1
}

classes = {v: k for k, v in label.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("error")
        break

    # Convert the entire frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to 150x150 (the expected input size of your model)
    processed_frame = cv2.resize(gray, (200, 200))
    processed_frame = processed_frame.astype(np.float32) / 255.0
    processed_frame = processed_frame.reshape(1, 200, 200, 1)

    # Predict gender
    prediction = model.predict(processed_frame)
    pred_class = np.argmax(prediction)
    gender = classes[pred_class]

    # Display prediction on the webcam window
    cv2.putText(
        frame,
        f"Prediction: {gender}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.imshow('Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()