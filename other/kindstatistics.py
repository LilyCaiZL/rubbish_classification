import xml.etree.ElementTree as ET
import os

TRAIN_STATISTICS = {
    'plastic': [0],
    'metal': [0],
    'paper': [0],
    'glass': [0]
}

all_filepath = []
for dirpath, dirname, filepath in os.walk(r"D:\document\aistudio\SSD_PaddlePaddle\dataset\train\image_xml"):
    for one_file in filepath:
        one_file_path = os.path.join(dirpath, one_file)
        all_filepath.append(one_file_path)

def _process_image(path):
    for i in path:
        # Read the XML annotation file.
        # 3. 构造xml文件路径
        filename = i
        # 4. xml文件解析
        tree = ET.parse(filename)
        # a. 得到xml文件对应的根节点
        root = tree.getroot()
        # 有几种类别就设置几种FLAG
        FLAG1 = False
        FLAG2 = False
        FLAG3 = False
        FLAG4 = False
        for obj in root.findall('object'):
            label = obj.find('name').text
            try:
                # 有几种类别就设置几种FLAG
                if label == "plastic":
                    FLAG1 = True
                if label == "metal":
                    FLAG2 = True
                if label == "paper":
                    FLAG3 = True
                if label == "glass":
                    FLAG4 = True
            except:
                continue

        if FLAG1 == True:
            TRAIN_STATISTICS["plastic"][0] += 1
        if FLAG2 == True:
            TRAIN_STATISTICS["metal"][0] += 1
        if FLAG3 == True:
            TRAIN_STATISTICS["paper"][0] += 1
        if FLAG4 == True:
            TRAIN_STATISTICS["glass"][0] += 1

_process_image(all_filepath)
print(TRAIN_STATISTICS)
