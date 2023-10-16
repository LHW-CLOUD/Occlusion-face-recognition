
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
    det_face = dlib.get_frontal_face_detector()
    # 加载标志点检测器
    det_landmarks = dlib.shape_predictor("./facemask/dat/shape_predictor_68_face_landmarks.dat")  # 68点
    out_dir ="./LFW-MASK-new"                 #     "./LFW-MASK"
    def walkFile(file):
            roots=os.listdir(file)    #【123】
            for i in roots:   #123
                names=os.listdir("./lfw-112X96/{}".format(i))   #./lfw-112X96/{}
                str="./lfw-112X96/{}".format(i)   #./lfw-112X96/{}
                for J in names:
                    image_path = os.path.join(str, J)  #./CASIA-WebFace/0000045\011.jpg

                    img = cv2.imread(image_path)         #读取图片
                    img = resize(img, width=500)         #把图片resize一下
                    num=random.choice(range(5))
                    if num == 0:
                        deal = Image.open("./墨镜图片/777.png")       #读取眼镜图片
                    elif num == 1:
                        deal = Image.open("./墨镜图片/glasses2.png")
                    elif num==2:
                        deal = Image.open("./墨镜图片/111.png")
                    elif num==3:
                        deal = Image.open("./墨镜图片/222.png")

                    else:
                        deal = Image.open("./墨镜图片/999.png")
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #转化为灰度图
                    rects = det_face(img_gray, 0)
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    for rect in rects:
                        face = {}
                        shades_width = rect.right() - rect.left()  # 人脸框的长度

                        # 用于检测当前人脸所在位置方向的预测器
                        shape = det_landmarks(img_gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # 从输入图像中抓取每只眼睛的轮廓
                        leftEye = shape[36:42]
                        rightEye = shape[42:48]

                        # 计算每只眼睛的中心
                        leftEyeCenter = leftEye.mean(axis=0).astype("int")
                        rightEyeCenter = rightEye.mean(axis=0).astype("int")

                        # 计算眼心之间的夹角
                        dY = leftEyeCenter[1] - rightEyeCenter[1]
                        dX = leftEyeCenter[0] - rightEyeCenter[0]
                        angle = np.rad2deg(np.arctan2(dY, dX))  # 旋转用的角度

                        # 图片重写
                        current_deal = deal.resize((shades_width, int(shades_width * deal.size[1] / deal.size[0])),
                                                   # 调整眼镜图片
                                                   resample=Image.Resampling.LANCZOS)
                        current_deal = current_deal.rotate(angle, expand=True)  # 旋转眼镜图片
                        current_deal = current_deal.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        #    current_deal.show()
                        face['glasses_image'] = current_deal
                        left_eye_x = leftEye[0, 0] - shades_width // 4
                        left_eye_y = leftEye[0, 1] - shades_width // 6
                        face['final_pos'] = (left_eye_x, left_eye_y)

                        img.paste(current_deal, (left_eye_x, left_eye_y), current_deal)  # 调节眼镜位置    PIL格式

                    image_id = os.path.basename(image_path).split(".")[0]
                    out_dir = "./LFW-MASK-new/{}/".format(i)

                    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    opencv_image = cv2.resize( opencv_image,(96,112))
                    # 保存图片
                    # cv2.imshow("123",opencv_image)
                    # cv2.waitKey(0)
                    facedetector.save_image(opencv_image, out_dir, image_id)
                    print(out_dir)

walkFile("./lfw-112X96")  #./lfw-112X96
