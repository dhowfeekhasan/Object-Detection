# Import the libraries we need
import os
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns

# --- Define Paths for Kaggle (This is still correct) ---
DATA_PATH = '/kaggle/input/vehicle-detection-dataset/train/Final Train Dataset/'
ANNOTATIONS_PATH = DATA_PATH
IMAGES_PATH = DATA_PATH

# --- A function to read one XML file (UPDATED with error handling) ---
def parse_xml_file(xml_file):
    # We add a try...except block to handle broken files
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
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            objects_in_image.append({
                'filename': image_name,
                'class': class_name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
        return objects_in_image
    # If a ParseError happens, we catch it here
    except ET.ParseError:
        # Print a warning so we know which file was bad
        print(f"WARNING: Could not parse {xml_file}. The file is likely empty or corrupt. Skipping this file.")
        # Return an empty list for the broken file so our program doesn't crash
        return []

# --- Loop through all XMLs and build the DataFrame (This code is now safe) ---
xml_files = [os.path.join(ANNOTATIONS_PATH, f) for f in os.listdir(ANNOTATIONS_PATH) if f.endswith('.xml')]

all_objects_df = pd.concat([pd.DataFrame(parse_xml_file(f)) for f in xml_files], ignore_index=True)

# Let's see what we've got!
print("\nSuccessfully read all valid annotations from the correct folder!")
print(f"Found {len(all_objects_df)} objects in total.")
print("\nHere are the first 5 rows of our data table:")
print(all_objects_df.head())

# --- Visualize the class distribution (This stays the same) ---
plt.figure(figsize=(10, 6))
sns.countplot(data=all_objects_df, y='class', order=all_objects_df['class'].value_counts().index)
plt.title('How Many Objects of Each Class?')
plt.xlabel('Number of Objects')
plt.ylabel('Vehicle Class')
plt.show()
