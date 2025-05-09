import numpy as np
from PIL import Image, ImageDraw


def create_rectangle(img_width, img_height, rect_width, rect_height, outline_thickness, background_color, rect_color):
    """
    Create an image with a rectangle drawn on it.

    Parameters:
    img_width (int): Width of the image.
    img_height (int): Height of the image.
    rect_width (int): Width of the rectangle.
    rect_height (int): Height of the rectangle.
    outline_thickness (int): Thickness of the rectangle outline.
    background_color (tuple): Background color of the image (R, G, B).
    rect_color (tuple): Color of the rectangle (R, G, B).

    Returns:
    Image: An image with a rectangle drawn on it.
    """
    img = Image.new('RGB', (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, rect_width, rect_height],
                   outline=rect_color, width=outline_thickness)
    return img


def translate(image, tx, ty):
    width, height = image.size

    pixels = np.array(image)
    translated_pixels = np.zeros_like(pixels) + 255

    for y in range(height):
        for x in range(width):
            new_x, new_y = x + tx, y + ty
            if 0 <= new_x < width and 0 <= new_y < height:
                translated_pixels[new_y, new_x] = pixels[y, x]

    return Image.fromarray(translated_pixels)


def rotate(image, angle):
    return image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def shear(image, sx, sy):
    width, height = image.size
    matrix = [1, sx, 0, sy, 1, 0]
    return image.transform((width, height), Image.AFFINE, matrix, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


# Rectangle
img = create_rectangle(600, 400, 500, 300, 5, (255, 255, 255), 'red')
img.save('../results/ex1b_original.jpg')

img_1 = translate(img.copy(), 50, 30)
img_1.save('../results/ex1b_translate_50_30.jpg')

img_2 = rotate(img.copy(), 30)
img_2.save('../results/ex1b_rotate_30.jpg')

img_3 = shear(img.copy(), 0.5, 0)
img_3.save('../results/ex1b_shear_x_0_5.jpg')

img_4 = shear(img.copy(), 0, 0.5)
img_4.save('../results/ex1b_shear_y_0_5.jpg')
