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
input_path=r"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\part2\\"
shuxing=["Thetas","Thetar","Lamba","Thetas","Alpha","Ksaturated"]
for i in shuxing:
    img1=crop_image(input_path+f"re_{i}0_5.jpg", 50, 350, 9040, 9150)

    img2=crop_image(input_path+f"re_{i}_layer2.jpg", 820, 350, 9040, 9150)
    #
    img3=crop_image(input_path+f"re_{i}_layer3.jpg", 50, 620, 9780, 9150-350+620-300)
    target=Image.new('RGB',(9040+9040+100,9150+9150-350+620-300+100),"white")
    target.paste(img1,(0,0))
    target.paste(img2, (9040+100, 0))
    target.paste(img3, (int((9150+9150-350+620-300+100)/2-(9780)/2), 9150+100))
    target.show()
    target.save(fr"F:\bsPaper\SDG&hyc_pic2\SDG&hyc_pic2\soil\clip_2\{i}.jpg")

