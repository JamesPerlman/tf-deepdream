from PIL import Image, ImageChops

def blended(img_in, img_out, alpha, scale, translate):
    image1 = Image.open(img_in).convert("RGBA")
    image2 = Image.open(img_out).convert("RGBA")
    image3 = Image.open(img_out).convert("RGBA")

    w, h = image1.size
    
    # alpha-blend the images
    sx, sy = (1, 1) #np.divide(scale, 100)
    tx, ty = map(int, translate)

    image2 = image2.resize((int(w * sx), int(h * sy)))
    image3.paste(image2, (-tx - int(w * (1 - 1 / sx) / 2), -ty - int(h * (1 - 1 / sy) / 2)))
    image2 = image3

    alphaBlended = ImageChops.blend(image1, image2, alpha=alpha)
    rgb_img = alphaBlended.convert('RGB')

    return rgb_img
