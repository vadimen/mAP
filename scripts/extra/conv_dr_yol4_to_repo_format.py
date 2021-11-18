import os
import cv2
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
# get result.txt "./darknet detector test data/obj.data data/yolov4-custom.cfg backup/yolov4-custom_best.weights -dont_show -ext_output < data/test.txt > result.txt"
parser.add_argument("--results-file", type=str, help="result.txt file made with detector test cmd")
parser.add_argument("--dr-folder", type=str, help="folder to save resulted txts")
parser.add_argument('--classes', type=str, help='names of classes as "name1,name2" in order as in labels.txt ')
args = parser.parse_args()

def convert():
    IN_FILE = args.results_file

    # create folder to save
    os.makedirs(args.dr_folder, exist_ok=True)

    PATTERN = ": Predicted in "
    SEPARATOR_KEY = "(\\.jp.?g): Predicted in "
    classes = args.classes.split(',')

    outfile = None
    with open(IN_FILE) as infile:
        for line in infile:
            if PATTERN in line:
                # get text between two substrings (SEPARATOR_KEY and '/' symbol)
                image_path = re.search(".*/(.*)" + SEPARATOR_KEY, line)
                # get the image name (the final component of a image_path)
                # e.g., from 'data/horses_1' to 'horses_1'
                image_name = os.path.basename(image_path.group(1))
                # close the previous file
                if outfile is not None:
                    outfile.close()
                # open a new file
                outfile = open(os.path.join(args.dr_folder, image_name + '.txt'), 'w')
            elif (outfile is not None) and ('Detection layer:' not in line) and ("Enter Image Path:" not in line):
                # split line on first occurrence of the character ':' and '%'
                class_name, info = line.split(':', 1)
                # print("-->", info)
                confidence, bbox = info.split('%', 1)
                # get all the coordinates of the bounding box
                bbox = bbox.replace(')', '')  # remove the character ')'
                # go through each of the parts of the string and check if it is a digit
                left, top, width, height = [int(s) for s in bbox.split() if s.lstrip('-').isdigit()]
                right = left + width
                bottom = top + height
                outfile.write("{} {} {} {} {} {}\n".format(classes.index(class_name), confidence, left, top, right, bottom))

convert()