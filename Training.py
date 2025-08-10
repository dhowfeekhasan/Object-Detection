import os
os.environ["KERAS_BACKEND"] = "tensorflow"
# Disable XLA to avoid RaggedTensor compilation issues
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras_cv
import numpy as np

# Disable XLA compilation and mixed precision
tf.config.optimizer.set_jit(False)
tf.keras.mixed_precision.set_global_policy('float32')

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --- Section A: Load Data from Files ---
# (This section remains unchanged)
print("Loading data...")
DATA_PATH = '/kaggle/input/vehicle-detection-dataset/train/Final Train Dataset/'
IMAGES_PATH = DATA_PATH
ANNOTATIONS_PATH = DATA_PATH

def parse_xml_file(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_name = root.find('filename').text
        objects_in_image = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in ['car', 'bus', 'motorbike', 'auto']:
                continue
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text); ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text); ymax = int(box.find('ymax').text)
            objects_in_image.append({
                'filename': image_name, 'class': class_name,
                'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
            })
        return objects_in_image
    except ET.ParseError:
        return []

xml_files = [os.path.join(ANNOTATIONS_PATH, f) for f in os.listdir(ANNOTATIONS_PATH) if f.endswith('.xml')]
all_objects_df = pd.concat([pd.DataFrame(parse_xml_file(f)) for f in xml_files], ignore_index=True)
print(f"Successfully loaded {len(all_objects_df)} objects.")


# --- Section B: Prepare Data with Fixed Tensor Handling ---
# (This section remains unchanged)
print("\nPreparing data loaders...")
class_names = sorted(all_objects_df['class'].unique())
class_mapping = {name: i for i, name in enumerate(class_names)}
all_objects_df['class_id'] = all_objects_df['class'].map(class_mapping)
# ... (rest of data prep code is the same) ...
# Get all available image files with their exact paths
available_images = {}
for file in os.listdir(IMAGES_PATH):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        full_path = os.path.join(IMAGES_PATH, file)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            available_images[file] = full_path

# Robust file matching
data = {}
for _, row in all_objects_df.iterrows():
    filename = row['filename']
    full_path = available_images.get(filename)
    if full_path:
        if full_path not in data:
            data[full_path] = {'boxes': [], 'classes': []}
        data[full_path]['boxes'].append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        data[full_path]['classes'].append(row['class_id'])

image_paths = list(data.keys())
bounding_boxes_list = [data[path] for path in image_paths]
train_paths, val_paths, train_boxes, val_boxes = train_test_split(
    image_paths, bounding_boxes_list, test_size=0.2, random_state=42
)

BATCH_SIZE = 4
TARGET_SIZE = (640, 640)
MAX_DETECTIONS = 100

def create_fixed_dataset(image_paths, bounding_boxes_list):
    def pad_to_fixed_size(boxes, classes, max_detections):
        num_boxes = len(boxes)
        if num_boxes > max_detections:
            boxes = boxes[:max_detections]
            classes = classes[:max_detections]
            num_boxes = max_detections
        padded_boxes = np.zeros((max_detections, 4), dtype=np.float32)
        padded_classes = np.zeros((max_detections,), dtype=np.float32)
        if num_boxes > 0:
            padded_boxes[:num_boxes] = boxes
            padded_classes[:num_boxes] = classes
        return padded_boxes, padded_classes
    
    def safe_load_image(image_path, boxes, classes):
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_raw, channels=3, expand_animations=False)
        image = tf.ensure_shape(image, [None, None, 3])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize_with_pad(image, TARGET_SIZE[0], TARGET_SIZE[1])
        return image, boxes, classes
    
    processed_data = []
    for img_path, bbox_data in zip(image_paths, bounding_boxes_list):
        boxes = np.array(bbox_data['boxes'], dtype=np.float32)
        classes = np.array(bbox_data['classes'], dtype=np.float32)
        padded_boxes, padded_classes = pad_to_fixed_size(boxes, classes, MAX_DETECTIONS)
        processed_data.append((img_path, padded_boxes, padded_classes))
    
    image_paths_tensor = [item[0] for item in processed_data]
    boxes_tensor = np.stack([item[1] for item in processed_data])
    classes_tensor = np.stack([item[2] for item in processed_data])
    
    dataset = tf.data.Dataset.from_tensor_slices((
        image_paths_tensor,
        tf.constant(boxes_tensor, dtype=tf.float32),
        tf.constant(classes_tensor, dtype=tf.float32)
    ))
    dataset = dataset.map(safe_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    def format_for_keras_cv(image, boxes, classes):
        return {"images": image, "bounding_boxes": {"boxes": boxes, "classes": classes}}
        
    dataset = dataset.map(format_for_keras_cv, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_fixed_dataset(train_paths, train_boxes)
val_ds = create_fixed_dataset(val_paths, val_boxes)
print("Data loaders are ready.")


# --- Section C: Build and Train the YOLOv8 Model ---
print("\nBuilding YOLOv8 model...")

# --- CHANGE 1: Define the path for our resuming checkpoint ---
checkpoint_path = '/kaggle/working/yolov8_latest_checkpoint.keras'

model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_backbone_coco",
    num_classes=len(class_mapping),
    bounding_box_format="xyxy"
)

# --- CHANGE 2: Check if a checkpoint exists and load it ---
if os.path.exists(checkpoint_path):
    print("--- Checkpoint found, resuming training from the last saved state. ---")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'YOLOV8Detector': keras_cv.models.YOLOV8Detector}
    )
else:
    print("--- No checkpoint found, starting training from scratch. ---")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    classification_loss='binary_crossentropy',
    box_loss='ciou',
    jit_compile=False
)
print("Model compiled successfully.")

# --- CHANGE 3: Update the callbacks ---
callbacks = [
    # This callback still saves only the BEST model
    tf.keras.callbacks.ModelCheckpoint(
        '/kaggle/working/yolov8_best.keras',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    ),
    # This NEW callback saves the model state after EVERY epoch for resuming
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, # Saves to our new path
        save_best_only=False, # Important: save regardless of performance
        verbose=0
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)
print("\nTraining completed successfully!")
