import os
import struct

data_root = "src/numberOCR/data"

# 加载图片并转换为灰度值的二维列表
def load_bmp(path):
    with open(path,"rb") as f:
        data = f.read()

    width = struct.unpack_from("<I", data, 18)[0]
    height = struct.unpack_from("<I", data, 22)[0]
    
    offset = struct.unpack_from("<I", data, 10)[0]

    pixels = []
    
    row_padded = (width * 3 + 3) & ~3

    for y in range(height):
        row = []
        for x in range(width):
            i = offset + y * row_padded + x * 3
            b, g, r = data[i], data[i+1], data[i+2]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            row.append(gray/255.0)
        pixels.append(row)
    return pixels

# 反转灰度并合并成一维列表
def flatten(pixels):
    flat = []
    for row in pixels:
        for val in row:
            flat.append(1.0 - val)
    return flat

# 处理Bmp
def process_bmp(path):
    img = load_bmp(path)
    img.reverse()
    return flatten(img)

# 处理训练集
def load_data(dataset):
    for label_name in os.listdir(data_root):
        label_dir = os.path.join(data_root, label_name)
        if os.path.isdir(label_dir):
            try:
                label = int(label_name)
            except ValueError:
                print(f"Skipping non-integer label: {label_name}")
                continue
                
            for filename in os.listdir(label_dir):
                if filename.endswith(".bmp"):
                    file_path = os.path.join(label_dir, filename)
                    try:
                        features = process_bmp(file_path)
                        dataset.append((features, label))
                        print(f"Loaded {file_path} with label {label}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        
    print(f"Total samples loaded: {len(dataset)}")
            