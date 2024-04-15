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
files=["0_5","_layer2","_layer3"]
for file in files:
    img1=crop_image(fr"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\part2\re_Thetas{file}.jpg", 50, 350, 9040, 9150)

    # img2=crop_image(rf"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\part2\re_Thetar{file}.jpg", 820, 350, 9040, 9150)
    img2=crop_image(rf"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\part2\re_Lamba{file}.jpg", 820, 350, 9040, 9150)
    img3=crop_image(rf"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\part2\re_Alpha{file}.jpg", 50, 350+270, 9040, 9150-350+620-300)
    img4=crop_image(rf"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\part2\re_Ksaturated{file}.jpg", 820, 620, 9040,9150-350+620-300)
    # img6=crop_image(r"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\part2\\"+files[1], 820, 620, 9040, 9150-350+620-300)
    #
    target=Image.new("RGB",(9040*2+100,9150+(9150-350+620-300)+100),"white")
    target.paste(img1,(0,0))
    target.paste(img2, (9040+100, 0))
    target.paste(img3,(0,9150+100))
    target.paste(img4, (9040+100, 9150+100))
    # target.paste(img5,(int((9040*2+100)/2-9040/2),9150+100+9150-270+100))
    # target.paste(img6, (9040+100, 9150+100+9150-270+100))

    target.show()
    target.save(fr"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\clip_2\{file}.jpg")

