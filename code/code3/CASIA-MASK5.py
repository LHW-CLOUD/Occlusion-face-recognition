###########################resize图片的大小
import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "libs")
from tqdm import tqdm
from facemask.wearmask import FaceMaskCreator
from pybaseutils import file_utils, image_utils
import cv2



import cv2
import numpy as np
import dlib
from PIL import Image, ImageDraw, ImageFont
from imutils import face_utils, translate, rotate, resize
from facemask.wearmask import FaceMaskCreator
import random
facedetector=FaceMaskCreator()



if __name__ == '__main__':
    random.seed(100)
    # 创建人脸检测器

    # 加载标志点检测器

    out_dir ="H://new_lfw-blur//"                 #     "./LFW-MASK"
    def walkFile(file):
            roots=os.listdir(file)    #【123】
            for i in roots:   #123
                names=os.listdir("H://lfw-112X96//{}".format(i))   #./lfw-112X96/{}
                str="H://lfw-112X96//{}".format(i)   #./lfw-112X96/{}
                for J in names:
                    image_path = os.path.join(str, J)

                    img = cv2.imread(image_path)         #读取图片
                    blur_radius=10
                    blur_image=cv2.GaussianBlur(img,(9,9),18)

                    image_id = os.path.basename(image_path).split(".")[0]
                    out_dir = "H://new_lfw-blur//{}/".format(i)
                #    opencv_image = cv2.resize( img ,(96,112),interpolation=cv2.INTER_AREA)
                    # 保存图片
                    # cv2.imshow("123",opencv_image)
                    # cv2.waitKey(0)
                    facedetector.save_image(blur_image, out_dir, image_id)
                 #   print(out_dir)

walkFile("H://lfw-112X96")  #./lfw-112X96


#######