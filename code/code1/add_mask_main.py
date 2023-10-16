import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "libs")
from tqdm import tqdm
from facemask.wearmask import FaceMaskCreator
from pybaseutils import file_utils, image_utils
import cv2
import numpy as np
if __name__ == '__main__':
    facedetector=FaceMaskCreator()
    w=0
    out_dir ="H:\\LFW-MASK"                 #     "./LFW-MASK"
    def walkFile(file):
            global w
            roots=os.listdir(file)    #【123】
      #      print(roots)
            for i in roots:   #123

                names=os.listdir(".//112_112//{}".format(i))   #./lfw-112X96/{}
                str=".//112_112//{}".format(i)   #./lfw-112X96/{}
                for J in names:
                #    w = w + 1
                    image_path = os.path.join(str, J)
              #      print(image_path)    #./CASIA-WebFace/0000045\011.jpg
                    image = image_utils.read_image(image_path, use_rgb=True)   #112 96
           #         cv2.imshow("111",image)
                #    if(w%2==0):
                    mask = facedetector.create_masks(image, mask_type="random", vis=False)  # 随机挑选口罩模板   返回的是佩戴了mask的图片
                #    cv2.imshow('222',mask)

                #    mask=image          #返回的是原图
                    my=mask-image
                    new_img = cv2.cvtColor(my, cv2.COLOR_RGB2GRAY)    #变为灰度图像
             #       cv2.imshow("444", my)
                    new_img_1=cv2.multiply(new_img,255)
            #        new_img_1=new_img_1/255   ###1
                    new_img_1=np.expand_dims(new_img_1,axis=2)   ###掩膜mask   先扩展通道

                    new_img_1 = np.concatenate((new_img_1, new_img_1, new_img_1), axis=2)    #拼接mask，编成三通道
                    mask2=(mask*new_img_1).astype("uint8")      ###

                # ------------------------------------------------#
                #   将新图片转换成Image的形式
                # ------------------------------------------------#

         #           cv2.imshow("555", new_img_1)
            #        cv2.imshow("333",new_img)
            #        cv2.waitKey(0)
                 #   else:
                  #      mask=image          #返回的是原图
                    image_id = os.path.basename(image_path).split(".")[0]
                    out_dir1 = "H:\\LFW-MASK\\{}\\".format(i)
                    out_dir2 = "H:\\LFW-MASK_MASK\\{}\\".format(i)
                    #facedetector.save_image(image, mask, face_rects, out_dir, image_id)
                    facedetector.save_image(mask, out_dir1, image_id)      #保存添加mask图片
                    facedetector.save_image2(new_img_1, out_dir2, image_id)    #保存添加口罩的照片
walkFile(".//112_112")  #./lfw-112X96