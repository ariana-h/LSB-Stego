from PIL import Image
import os
import numpy as np

def decode(input_path):
    with Image.open(input_path) as img:
        img = Image.open(input_path, 'r')
        array = np.array(list(img.getdata()))

        if img.mode == 'RGB':
            n = 3
        elif img.mode == 'RGBA':
            n = 4
        total_pixels = array.size//n

        hidden_bits = ""
        for p in range(total_pixels):
            for q in range(0, 3):
                hidden_bits += (bin(array[p][q])[2:][-1])

        hidden_bits = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]

        message = ""
        for i in range(len(hidden_bits)):
            if message[-8:] == "pr34mb13":
                break
            else:
                message += chr(int(hidden_bits[i], 2))
        if "$t3g0" in message:
            print("Hidden Message:", message[:-8])
        else:
            print("No Hidden Message Found")            

def main():
    src = "encoded"
    for file in os.listdir(src):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
            input_path = os.path.join(src, file) 
            decode(input_path)


if __name__ == "__main__":
    main()