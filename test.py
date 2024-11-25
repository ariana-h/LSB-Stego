import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm 

total_pos, total_neg, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
y_true = [] 
y_pred = [] 

model = tf.keras.models.load_model('best_model.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    return image

def log(message):
    with open("output_log.txt", "a") as file:
        file.write(message + "\n")

def detect_message(image_path, is_stego):
    global total_pos, total_neg, tp, fp, tn, fn, y_true, y_pred

    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    prediction = model.predict(preprocessed_image, verbose=0)
    y_pred.append(prediction[0][0])
    y_true.append(1 if is_stego else 0)  # 1 for stego, 0 for non-stego

    if prediction[0][0] > 0.5:
        log(f"Steganography detected in {image_path}")
        total_pos += 1
        if not is_stego:
            fp += 1  # False positive
            log(f"False positive: {fp} / {total_pos}")
        else:
            tp += 1  # True positive
            log(f"True positive: {tp} / {total_pos}")
        decode(image_path)
    else:
        log(f"No message detected in {image_path}")
        total_neg += 1
        if not is_stego:
            tn += 1  # True negative
            log(f"True negative: {tn} / {total_neg}")
        else:
            fn += 1  # False negative
            log(f"False negative: {fn} / {total_neg}")
            decode(image_path)

def decode(input_path):
    with Image.open(input_path) as img:
        array = np.array(list(img.getdata()))
        total_pixels = array.size // 3  # Assuming img.mode == 'RGB'
        hidden_bits = ""
        for p in range(total_pixels):
            for q in range(3):
                hidden_bits += bin(array[p][q])[-1]
        hidden_bytes = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]
        message = ""
        for byte in hidden_bytes:
            if message[-8:] == "pr34mb13":
                break
            else:
                message += chr(int(byte, 2))
        if "pr34mb13" in message:
            log(f"Hidden Message: {message[:-8]}")
        else:
            log("No Hidden Message Found")

def collect_files(folder_path, is_stego):
    return [(os.path.join(folder_path, f), is_stego) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def main():
    stego_folder = "dataset/test/stego"
    non_stego_folder = "dataset/test/non-stego"

    if os.path.exists("output_log.txt"):
        os.remove("output_log.txt")
        
    files = collect_files(stego_folder, is_stego=True) + collect_files(non_stego_folder, is_stego=False)

    print("Detecting steganography in images...")
    log("Detecting steganography in images...")

    # Process all images with a single loading bar
    for file_path, is_stego in tqdm(files, desc="Processing images"):
        detect_message(file_path, is_stego)

    accuracy = (tp + tn) / (total_pos + total_neg)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)


    print("==========================================================")
    print(f"True positive: {tp / total_pos:.4f}")
    print(f"False positive: {fp / total_pos:.4f}")
    print(f"True negative: {tn / total_neg:.4f}")
    print(f"False negative: {fn / total_neg:.4f}")

    print("----------------------------------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

   
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

if __name__ == "__main__":
    main()
