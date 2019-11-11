import sys
import os
import json
import xml.etree.ElementTree as ET


START_BOUNDING_BOX_ID = 1
# PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
# PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#                           "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#                           "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#                           "motorbike": 14, "person": 15, "pottedplant": 16,
#                           "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
PRE_DEFINE_CATEGORIES = {"glass": 1, "paper": 2, "metal": 3, "plastic": 4}
def gen_xml_file(xml_dir,xml_list):
    '''generate xml list file'''
    import os
    root = xml_dir
    with open(xml_list,'a+') as f:
        for file in os.listdir(root):
            f.write(file)
            f.write('\n')

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = len(filename)
        return filename
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    start_index = 20190000000
    for line in list_fp:
        start_index += 1
        line = line.strip()
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')

        #if len(path) == 1:
         #   filename = os.path.basename(path[0].text)
        #elif len(path) == 0:
        if get_and_check(root, 'filename', 1).text[0]=='m' or get_and_check(root, 'filename', 1).text[0] == 'g' or get_and_check(root, 'filename', 1).text[0] == 'p':
            filename = get_and_check(root, 'filename', 1).text
        else:
            filename = get_and_check(root, 'filename', 1).text + ".jpg"
        #else:
         #   raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        image_id = start_index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
         # segmented = get_and_check(root, 'segmented', 1).text
         # assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)

            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':
    # import os
    # filePath = 'E:\\mypython\\pythonpure\\pascalToCoco\\image_xml'
    # print(os.listdir(filePath))
    #
    # with open('t.txt','a+') as f:
    #     for i in os.listdir(filePath):
    #         f.writelines(i+'\n')
    xml_dir = 'D:\\document\\aistudio\\SSD_PaddlePaddle\\dataset\\train\\image_xml'
    xml_list = 'train.txt'
    json_file = 'instances_train2014.json'
    ## generate xml_list file
    gen_xml_file(xml_dir=xml_dir,xml_list=xml_list)
    ## then process xml file to coco json file
    convert(xml_list = xml_list, xml_dir = xml_dir,json_file=json_file)