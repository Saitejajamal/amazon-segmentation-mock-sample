# cheque_amount_reader.py

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from PIL import Image
import pytesseract

# -----------------------------
# Build CNN model
# -----------------------------
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn_model()

# -----------------------------
# Load MNIST and train CNN
# -----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

st.sidebar.title("CNN Training Options")
if st.sidebar.button("Train CNN on MNIST"):
    st.write("Training CNN on MNIST dataset...")
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    loss, acc = model.evaluate(x_test, y_test)
    st.success(f"Model trained! Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Handwritten Cheque Amount Reader")
st.write("Upload a cheque image and the model will read the handwritten amount.")

uploaded_file = st.file_uploader("Upload cheque image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Cheque", use_column_width=True)

    if st.button("Read Cheque Amount"):
        # Convert to OpenCV
        img_cv = np.array(img.convert('RGB'))[..., ::-1]

        # -----------------------------
        # Preprocessing
        # -----------------------------
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # -----------------------------
        # Manual crop of amount box (for demo, you can adjust coordinates)
        # -----------------------------
        h, w = thresh.shape
        # Example: crop bottom-right quarter (adjust according to cheque)
        amount_region = thresh[int(h*0.5):h, int(w*0.5):w]

        # -----------------------------
        # OCR for decimal and text
        # -----------------------------
        amount_text = pytesseract.image_to_string(amount_region, config='--psm 7 -c tessedit_char_whitelist=0123456789.')
        amount_text = amount_text.strip().replace(' ','')

        # -----------------------------
        # Digit segmentation
        # -----------------------------
        contours, _ = cv2.findContours(amount_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_boxes = []
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            if w_box>5 and h_box>10:
                digit_boxes.append((x,y,w_box,h_box))
        digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

        digits = ""
        for box in digit_boxes:
            x, y, w_box, h_box = box
            digit_img = amount_region[y:y+h_box, x:x+w_box]
            digit_img_resized = cv2.resize(digit_img, (28,28))
            digit_img_normalized = digit_img_resized/255.0
            digit_img_reshaped = digit_img_normalized.reshape(1,28,28,1)
            pred = model.predict(digit_img_reshaped)
            digits += str(pred.argmax())

        # Combine CNN digits and OCR decimal
        final_amount = digits
        if '.' in amount_text:
            final_amount += '.' + amount_text.split('.')[-1]

        st.success(f"Recognized Cheque Amount: ₹{final_amount}")
