import math
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

    x1 = (img_width - rect_width) // 2
    y1 = (img_height - rect_height) // 2
    x2 = x1 + rect_width
    y2 = y1 + rect_height

    draw.rectangle([x1, y1, x2, y2], outline=rect_color, width=outline_thickness)
    return img


def translate(image, tx, ty):
    """
    Apply translation transformation using the matrix:
    [1  0  tx]
    [0  1  ty]
    [0  0   1]
    """
    width, height = image.size
    pixels = np.array(image)
    translated_pixels = np.zeros_like(pixels) + 255

    for y in range(height):
        for x in range(width):
            new_x, new_y = x + tx, y + ty
            if 0 <= new_x < width and 0 <= new_y < height:
                translated_pixels[new_y, new_x] = pixels[y, x]

    return Image.fromarray(translated_pixels)


def rotate(image, angle_degrees):
    """
    Apply rotation transformation using the matrix:
    [cos θ  -sin θ  0]
    [sin θ   cos θ  0]
    [0       0      1]
    """
    angle_radians = math.radians(angle_degrees)
    width, height = image.size
    pixels = np.array(image)
    rotated_pixels = np.zeros_like(pixels) + 255

    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)

    center_x, center_y = width // 2, height // 2

    for y in range(height):
        for x in range(width):
            x_centered = x - center_x
            y_centered = y - center_y

            new_x = int(x_centered * cos_theta - y_centered * sin_theta + center_x)
            new_y = int(x_centered * sin_theta + y_centered * cos_theta + center_y)

            if 0 <= new_x < width and 0 <= new_y < height:
                rotated_pixels[new_y, new_x] = pixels[y, x]

    return Image.fromarray(rotated_pixels)


def shear(image, shx, shy):
    """
    Apply shear transformation using two matrices:
    X-shear: [1  shx  0]   Y-shear: [1   0   0]
             [0   1   0]            [shy  1   0]
             [0   0   1]            [0    0   1]

    Combined shear effect is achieved by applying both transformations.

    Parameters:
    image: PIL Image object
    shx: float, shear factor in x direction
    shy: float, shear factor in y direction
    """
    width, height = image.size
    pixels = np.array(image)
    sheared_pixels = np.zeros_like(pixels) + 255

    for y in range(height):
        for x in range(width):
            # Apply both x and y shear transformations
            new_x = int(x + shx * y)  # x' = x + shx * y
            new_y = int(y + shy * x)  # y' = shy * x + y

            if 0 <= new_x < width and 0 <= new_y < height:
                sheared_pixels[new_y, new_x] = pixels[y, x]

    return Image.fromarray(sheared_pixels)


print('ex1b...')

# Rectangle
img = create_rectangle(1000, 1000, 600, 400, 5, (255, 255, 255), 'red')
img.save('../results/ex1b_original.jpg')

img_1 = translate(img.copy(), 50, 30)
img_1.save('../results/ex1b_translate_50_30.jpg')

img_2 = rotate(img.copy(), 30)
img_2.save('../results/ex1b_rotate_30.jpg')

img_3 = shear(img.copy(), 0.5, 0)
img_3.save('../results/ex1b_shear_x_0_5.jpg')

img_4 = shear(img.copy(), 0, 0.5)
img_4.save('../results/ex1b_shear_y_0_5.jpg')
