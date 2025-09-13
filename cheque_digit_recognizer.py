import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import confusion_matrix, classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0  

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

model = Sequential([
    Flatten(input_shape=(28, 28)),              
    Dense(units=128, activation='relu'),        
    Dense(units=10, activation='softmax')       
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=5,
                    validation_split=0.1,
                    verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes))


cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Digit Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

errors = np.where(y_pred_classes != y_test)[0]
plt.figure(figsize=(10,4))
for i, idx in enumerate(errors[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx],cmap='gray')
    plt.title(f"True:{y_test[idx]}\nPred:{y_pred_classes[idx]}")
    plt.axis('off')

plt.suptitle("Examples of Wrong Predictions")
plt.show()

import graphviz
print("\nPipeline Steps:")
print("1. Data Acquisition -> 2. Data Preprocessing -> 3. Model Building")
print("4. Training -> 5. Evluation -> 6. Error Analysis -> 7.Presentation")
