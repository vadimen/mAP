#this is a file that converts to this repo format results produced by yolov5
#or ground truth for yolov4 or yolov5
#from this repo https://github.com/ultralytics/yolov5
#using the command
# python3 val.py --weights 'best.pt' --data 'dataset_v4.yaml' --save-txt --save-conf  --batch 8

#yolo5 dr or ground truth format is {xcent, ycent, width, height} all divided by h,w of the image
#this repo format is {xleft, ytop, xright, ybottom} or {XMin, YMin, XMax, YMax}

import os
import argparse
import re
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--img-dir", type=str, help="path to image folder")
parser.add_argument("--gt-dr-dir", type=str, help="path to detection results containing many txts")
parser.add_argument("--save-dir", type=str, help="path to save resulted txts")
parser.add_argument("--dt", action="store_true", help="if set then parse confidence from lines,"+
                                                               "conf is the 5th parameter")
args = parser.parse_args()

def convert():
    #create folder to save
    os.makedirs(args.save_dir, exist_ok=True)

    image_names = os.listdir(args.img_dir)
    for txt_file in os.listdir(args.gt_dr_dir):
        #find .txt and exclude it from txt_file
        res = re.search("(\\.txt)$", txt_file)
        if not res:
            print("wrong file", txt_file)
            break
        image_name_to_check = txt_file[: res.span()[0]]

        txt_has_match = None
        for name in image_names:
            #search for .jpg .jpeg .png and exclude it
            res = re.search("(\\.....?)$", name)
            if res:
                tmp_name = name[:res.span()[0]]
                if image_name_to_check == tmp_name:
                    txt_has_match = name
                    break

        if txt_has_match:
            h, w, _ = cv2.imread(f"{args.img_dir}/{txt_has_match}").shape
            file = open(f"{args.gt_dr_dir}/{txt_file}", "r")
            lines = file.readlines()
            file = open(f"{args.save_dir}/{txt_file}", "w")
            for line in lines:
                cls, xcent, ycent, width, height, conf = [float(e) for e in line.split(" ")]
                xcent, width = xcent*w, width*w
                ycent, height = ycent*h, height*h
                xleft = int(xcent - width/2)
                ytop = int(ycent - height/2)
                xright = int(xcent + width/2)
                ybottom = int(ycent + height/2)
                if args.dt:
                    file.write(f"{int(cls)} {conf} {xleft} {ytop} {xright} {ybottom}\n")
                else:
                    file.write(f"{int(cls)} {xleft} {ytop} {xright} {ybottom}\n")
            file.close()
        else:
            print("!!!!!txt with no match", txt_file)

convert()