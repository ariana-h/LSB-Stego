import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50

class SaveBestModel(Callback):
    def __init__(self, monitor='val_accuracy', filepath='best_model.h5', save_best_only=True):
        super(SaveBestModel, self).__init__()
        self.monitor = monitor
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        
        if val_acc is None or val_loss is None:
            print("Warning: Validation accuracy or validation loss is not available.")
            return
        
        if np.isnan(val_loss):
            print(f"\nEpoch {epoch + 1}: Validation loss is NaN! Stopping training.")
            self.model.stop_training = True
            return

        # Save model only if validation accuracy improves and validation loss decreases
        if val_acc > self.best_val_acc and val_loss < self.best_val_loss:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            print(f"\nEpoch {epoch + 1}: Validation Accuracy improved to {val_acc:.4f}, Validation Loss decreased to {val_loss:.4f}. Saving model...")
            self.model.save(self.filepath, overwrite=True)
            print("Model saved!")


def read_image(image_path):
    image = cv2.imread(image_path)
    return image

def load_data(data_dir):
    images = [] 
    labels = []
    for foldername in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, foldername)
        label = 1 if foldername == 'stego' else 0  # Assuming stego images are in 'stego' folder
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            images.append(read_image(image_path))
            labels.append(label)
    return np.array(images), np.array(labels)

data_dir = 'dataset/train'
images, labels = load_data(data_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def unet_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Downsampling (Encoder)
    conv1 = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    leaky_relu1 = layers.LeakyReLU(alpha=0.2)(conv1)
    conv1 = layers.Conv2D(32, (3, 3), padding='same')(leaky_relu1)
    leaky_relu2 = layers.LeakyReLU(alpha=0.2)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(leaky_relu2)
    
    conv2 = layers.Conv2D(64, (3, 3), padding='same')(pool1)
    leaky_relu3 = layers.LeakyReLU(alpha=0.2)(conv2)
    conv2 = layers.Conv2D(64, (3, 3), padding='same')(leaky_relu3)
    leaky_relu4 = layers.LeakyReLU(alpha=0.2)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(leaky_relu4)

    conv3 = layers.Conv2D(128, (3, 3), padding='same')(pool2)
    leaky_relu5 = layers.LeakyReLU(alpha=0.2)(conv3)
    conv3 = layers.Conv2D(128, (3, 3), padding='same')(leaky_relu5)
    leaky_relu6 = layers.LeakyReLU(alpha=0.2)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(leaky_relu6)

    # Bottleneck
    conv4 = layers.Conv2D(256, (3, 3), padding='same')(pool3)
    leaky_relu7 = layers.LeakyReLU(alpha=0.2)(conv4)
    conv4 = layers.Conv2D(256, (3, 3), padding='same')(leaky_relu7)
    leaky_relu8 = layers.LeakyReLU(alpha=0.2)(conv4)
    leaky_relu8 = layers.SpatialDropout2D(0.2)(leaky_relu8) 
    
    # Upsampling (Decoder)
    up5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(leaky_relu8)
    up5 = layers.concatenate([up5, conv3], axis=-1)
    conv5 = layers.Conv2D(64, (3, 3), padding='same')(up5)
    leaky_relu9 = layers.LeakyReLU(alpha=0.2)(conv5)
    conv5 = layers.Conv2D(64, (3, 3), padding='same')(leaky_relu9)
    leaky_relu10 = layers.LeakyReLU(alpha=0.2)(conv5)

    up6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(leaky_relu10)
    up6 = layers.concatenate([up6, conv2], axis=-1)
    conv6 = layers.Conv2D(32, (3, 3), padding='same')(up6)
    leaky_relu11 = layers.LeakyReLU(alpha=0.2)(conv6)
    conv6 = layers.Conv2D(32, (3, 3), padding='same')(leaky_relu11)
    leaky_relu12 = layers.LeakyReLU(alpha=0.2)(conv6)

    up7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(leaky_relu12)
    up7 = layers.concatenate([up7, conv1], axis=-1)
    conv7 = layers.Conv2D(16, (3, 3), padding='same')(up7)
    leaky_relu13 = layers.LeakyReLU(alpha=0.2)(conv7)
    conv7 = layers.Conv2D(16, (3, 3), padding='same')(leaky_relu13)
    leaky_relu14 = layers.LeakyReLU(alpha=0.2)(conv7)
    
    flatten = layers.Flatten()(leaky_relu14)
    flatten = layers.Dropout(0.4)(flatten)
    
    outputs = layers.Dense(1, activation='sigmoid')(flatten)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
model = unet_model(input_shape)

model.compile(optimizer=Adam(learning_rate=3e-4, decay=1e-4),
              loss= 'binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model.summary()

model_checkpoint = SaveBestModel(filepath='best_model.h5', save_best_only=True)


model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), callbacks=[model_checkpoint])

best_model = models.load_model('best_model.h5')

train_loss, train_acc, train_precision, train_recall = best_model.evaluate(X_train, y_train)

val_loss, val_acc, val_precision, val_recall = best_model.evaluate(X_test, y_test)
y_pred = best_model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred) 
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

print('========================================')
print(f'Training accuracy: {train_acc}')
print(f'Training precision: {train_precision}')
print(f'Training recall: {train_recall}')
print('----------------------------------------')
print(f'Validation accuracy: {val_acc}')
print(f'Validation precision: {val_precision}')
print(f'Validation recall: {val_recall}')
print(f'ROC AUC: {roc_auc}')

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()