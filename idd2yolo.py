''' This script will take the IDD detection directory as input
and create another directory named IDD_detection_yolo with custom classes.
This will convert the voc format to yolo format.

Make sure that you are in the same directory as the dataset while running this.
Usage - python yolov5/idd2yolo.py
It will create another directory IDD_detection_yolo which contains 
all the images, converted anootations, and train & val txt files.
'''

import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
import shutil
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ipdb

classes = ['motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'caravan']

def visualize(size, bbox, cls):
    ''' To visualize the bboxes'''

    print(bbox, cls)

    dw = 1./(size[0])
    dh = 1./(size[1])
    
    x = (bbox[0] + bbox[1])/2.0 - 1
    y = (bbox[2] + bbox[3])/2.0 - 1
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]

    img = plt.imread(imgpath)
    plt.plot(bbox[0], bbox[2], marker='o', color='black')
    plt.plot(bbox[1], bbox[3], marker='o', color='black')
    plt.plot(x, y, marker='o', color='white')

    plt.gca().add_patch(Rectangle((bbox[0], bbox[2]), w, h, fill=False))
    plt.imshow(img)
    plt.show()

def initialize(idd_root, idd_yolo_root):
    ''' Create the IDD Yolo directories.'''
    
    def ignore_files(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    try:
        shutil.copytree(idd_root, idd_yolo_root, ignore=ignore_files)
    except:
        pass
    os.rename(join(idd_yolo_root, 'JPEGImages'), join(idd_yolo_root, 'images'))
    os.rename(join(idd_yolo_root, 'Annotations'), join(idd_yolo_root, 'labels'))
    print("Directories Created")

def convert(size, box):
    ''' Converts voc bbox to yolo bbox.'''

    dw = 1./(size[0])
    dh = 1./(size[1])

    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1

    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x,y,w,h)

def convert_annotation(infile):
    ''' Converts voc file format to yolo file format.'''
    
    filename = infile.split('/')[-1]
    inxml = 'IDD_Detection/Annotations/%s.xml'%(infile)
    outtxt = 'IDD_detection_yolo/labels/%s.txt'%(infile)
    outfile = open(outtxt, 'w')
    flag = False
    
    tree = ET.parse(open(inxml))
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        flag = True
        # visualize((w, h), b, cls)
    outfile.close()
    if not flag:
        os.remove(outtxt)
    return flag

def main():
    idd_root = 'IDD_Detection'
    idd_yolo_root = 'IDD_detection_yolo'
    initialize(idd_root, idd_yolo_root)

    train_txt_path = join(idd_root, 'train.txt')
    val_txt_path = join(idd_root, 'val.txt')
    test_txt_path = join(idd_root, 'test.txt')

    all_txts = [train_txt_path, val_txt_path, test_txt_path]
    for txt in all_txts:
        
        file_type = txt.split('/')[-1].replace('.txt', '')
        txt_out = open(join(idd_yolo_root, file_type+'.txt'), 'w')

        image_ids = open(txt, 'r').read().strip().split()
        for image_id in tqdm(image_ids):
            try:
                check = convert_annotation(image_id)
                if check:
                    src = join(idd_root, 'JPEGImages', image_id+'.jpg')
                    dst = join(idd_yolo_root, 'images', image_id+'.jpg')
                    shutil.copy(src, dst)
                    dstpath = os.path.realpath(dst)
                    txt_out.write(dstpath + '\n')
            except:
                continue
        txt_out.close()

if __name__ == '__main__':
    main()
