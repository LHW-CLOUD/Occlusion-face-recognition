def add_masks_faces(self, img, face_landmark: dict, mask_image: ImageFile):  
    if mask_image is None:
        return img
    image = Image.fromarray(img)   #原图
    nose_bridge = face_landmark['nose_bridge']     #nose_bridge 表示鼻梁
    nose_pt = nose_bridge[len(nose_bridge) * 1 // 4]  
    nose_v = np.array(nose_pt)

    chin_pts = face_landmark['chin']   
    chin_len = len(chin_pts)
    chin_bottom_pt = chin_pts[chin_len // 2]   #表示脸底点 chin_bottom_point
    chin_bottom_v = np.array(chin_bottom_pt)
    chin_left_pt = chin_pts[chin_len // 8]      #表示脸左点 chin_right_point
    chin_right_pt = chin_pts[chin_len * 7 // 8]  #表示脸右点 chin_right_point


    width = mask_image.width   
    height = mask_image.height  
    width_ratio = 1.2
    new_height = max(1, int(np.linalg.norm(nose_v - chin_bottom_v)))

    # left
    mask_left_img = mask_image.crop((0, 0, width // 2, height)) 
    mask_left_width = self.get_distance_from_point_to_line(chin_left_pt, nose_pt, chin_bottom_pt)  
    mask_left_width = max(1, int(mask_left_width * width_ratio))
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))

    # right
    mask_right_img = mask_image.crop((width // 2, 0, width, height))
    mask_right_width = self.get_distance_from_point_to_line(chin_right_pt, nose_pt, chin_bottom_pt)
    mask_right_width = max(1, int(mask_right_width * width_ratio))
    mask_right_img = mask_right_img.resize((mask_right_width, new_height))

    # merge mask
    size = (mask_left_img.width + mask_right_img.width, new_height)
    mask_img = Image.new('RGBA', size)
    mask_img.paste(mask_left_img, (0, 0), mask_left_img)
    mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

    angle = np.arctan2(chin_bottom_pt[1] - nose_pt[1], chin_bottom_pt[0] - nose_pt[0])    
    angle = 90 - angle * 180 / np.pi
    rotated_mask_img = mask_img.rotate(angle, expand=True)  
    mask_img = rotated_mask_img
    # calculate mask location       计算中心位置
    center_x = (nose_pt[0] + chin_bottom_pt[0]) // 2
    center_y = (nose_pt[1] + chin_bottom_pt[1]) // 2

    offset = mask_img.width // 2 - mask_left_img.width  
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2
    # add mask
    image.paste(mask_img, (box_x, box_y), mask_img)

    img2 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) * 0
    image2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    image2.paste(mask_img, (box_x, box_y), mask_img)

    image = np.asarray(image)
    image2 = np.asarray(image2)
    return image
