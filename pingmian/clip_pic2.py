import os

from PIL import Image

def crop_image(image_path, x, y, width, height):
    # 打开图片
    image = Image.open(image_path)
    # 裁剪图片
    cropped_image = image.crop((x, y, x + width, y + height))
    # cropped_image.show()
    return cropped_image
    # 显示裁剪后的图片
    # cropped_image.show()
    # 保存裁剪后的图片
    # cropped_image.save('cropped_image.jpg')

# 调用示例
files=os.listdir(r"C:\Users\10741\Documents\WeChat Files\wxid_z9ck3528bpc222\FileStorage\File\2024-03\flood2\flood2\\")

img1=crop_image(r"C:\Users\10741\Documents\WeChat Files\wxid_z9ck3528bpc222\FileStorage\File\2024-03\flood2\flood2\\"+files[0], 50, 50, 5960, 6940)

img2=crop_image(r"C:\Users\10741\Documents\WeChat Files\wxid_z9ck3528bpc222\FileStorage\File\2024-03\flood2\flood2\\"+files[1], 540,50, 5960, 6960)
img3=crop_image(r"C:\Users\10741\Documents\WeChat Files\wxid_z9ck3528bpc222\FileStorage\File\2024-03\flood2\flood2\\"+files[2], 50, 220, 5960, 6770)

img4=crop_image(r"C:\Users\10741\Documents\WeChat Files\wxid_z9ck3528bpc222\FileStorage\File\2024-03\flood2\flood2\\"+files[3], 540,220, 5960, 6770)
img5=crop_image(r"C:\Users\10741\Documents\WeChat Files\wxid_z9ck3528bpc222\FileStorage\File\2024-03\flood2\flood2\\"+files[4], 50, 220, 5960, 6940)

img6=crop_image(r"C:\Users\10741\Documents\WeChat Files\wxid_z9ck3528bpc222\FileStorage\File\2024-03\flood2\flood2\\"+files[5], 540,220, 5960, 6940)

#
target=Image.new("RGB",(5960*2+100,6940+6770+6940+200),"white")
target.paste(img1,(0,0))
target.paste(img2, (5960+100, 0))
target.paste(img3,(0,6940+100))
target.paste(img4, (5960+100, 6940+100))
target.paste(img5,(0,6940+6770+200))
target.paste(img6, (5960+100, 6940+6770+200))

target.show()
target.save(fr"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\clip_2\jishui.jpg")

