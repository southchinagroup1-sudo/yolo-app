import xml.etree.ElementTree as ET
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  # 新增：导入进度条库

# ================= 配置区 =================
XML_FILE = "/home/ai/data_disk1/yolo/PJ1/hand and basket/annotations.xml"
IMG_DIR = "/home/ai/data_disk1/yolo/PJ1/hand and basket/images"
OUTPUT_DIR = "/home/ai/data_disk1/yolo/PJ1/yolo_dataset" 

FRAME_STEP = 1  # 抽帧间隔：1 表示全量保留
VAL_RATIO = 0.2  # 验证集比例 20%
# ==========================================

CLASS_MAP = {'hand': 0, 'basket': 1}

def convert_cvat_to_yolo():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    tree = ET.parse(XML_FILE)
    root = tree.getroot()
    
    meta = root.find('meta')
    orig_size = meta.find('original_size')
    img_w = int(orig_size.find('width').text)
    img_h = int(orig_size.find('height').text)

    frames_dict = {}
    for track in root.findall('track'):
        label = track.get('label')
        if label not in CLASS_MAP: continue
        cls_id = CLASS_MAP[label]
        
        for box in track.findall('box'):
            frame_id = int(box.get('frame'))
            if box.get('outside') == '1': continue
            
            xtl = float(box.get('xtl')); ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr')); ybr = float(box.get('ybr'))
            
            x_center = (xtl + xbr) / 2.0 / img_w
            y_center = (ytl + ybr) / 2.0 / img_h
            width = (xbr - xtl) / img_w
            height = (ybr - ytl) / img_h
            
            if frame_id not in frames_dict:
                frames_dict[frame_id] = []
            frames_dict[frame_id].append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    all_frames = sorted(frames_dict.keys())
    sampled_frames = [f for f in all_frames if f % FRAME_STEP == 0]
    
    print(f"总有效帧数: {len(all_frames)}, 抽帧后数量: {len(sampled_frames)}")
    print("开始复制与转换文件 (这可能需要几分钟，请等待进度条跑完)...")
    
    random.shuffle(sampled_frames)
    val_count = int(len(sampled_frames) * VAL_RATIO)
    val_frames = set(sampled_frames[:val_count])

    # 新增：使用 tqdm 包裹循环，显示进度条
    for frame_id in tqdm(sampled_frames, desc="处理进度", unit="张"):
        split = 'val' if frame_id in val_frames else 'train'
        
        img_name = f"frame_{frame_id:06d}.PNG"
        img_path = os.path.join(IMG_DIR, img_name)
        
        if not os.path.exists(img_path):
            continue

        dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, img_name)
        shutil.copy(img_path, dst_img_path)
        
        label_name = img_name.replace('.PNG', '.txt')
        dst_label_path = os.path.join(OUTPUT_DIR, 'labels', split, label_name)
        with open(dst_label_path, 'w') as f:
            f.write('\n'.join(frames_dict[frame_id]))

    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

names:
  0: hand
  1: basket
"""
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"\n转换完成！数据集已保存至: {OUTPUT_DIR}")
    print(f"训练集: {len(sampled_frames) - len(val_frames)} 张, 验证集: {len(val_frames)} 张")

if __name__ == '__main__':
    convert_cvat_to_yolo()