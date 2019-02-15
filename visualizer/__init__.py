import graphics
import cv2
import numpy as np

font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 20
font_thickness = 20
font_color = (0, 0, 0)
line_type = cv2.LINE_AA
margin = 50


def convert_token(token):
    if token == '\\sqrt':
        return "O"
    elif token == '\\theta':
        return "O"
    elif token == '\\int':
        return "f"
    elif token[0] == '\\':
        return token[1:]
    return token


def create_token_img(token):
    token = convert_token(token)
    text_size, baseline = cv2.getTextSize(token, font_face, font_scale, font_thickness)
    img = graphics.new_image(text_size[0] + 2 * margin, text_size[1] + baseline + 2 * margin)
    cv2.putText(img, token, (margin, text_size[1] + margin), font_face, font_scale, font_color, font_thickness,
                lineType=line_type)

    # Remove white rows
    img = img[~np.all(img == 255, axis=1).all(1)]

    # Remove white columns
    img = img[:, ~np.all(img == 255, axis=0).all(1)]

    return img


def print_token(image, token, bounding_box):
    minx, miny, maxx, maxy = bounding_box
    image_width = graphics.w(image)
    image_height = graphics.h(image)

    width = int((maxx - minx) * image_width)
    height = int((maxy - miny) * image_height)
    width = width if width != 0 else 1
    height = height if height != 0 else 1

    token_img = create_token_img(token)
    token_img = graphics.resize(token_img, width, height)

    x = int(minx * image_width)
    y = int(miny * image_height)

    graphics.paste(image, token_img, x, y)

    return image


def create_image(input, width=400, height=200):
    image = graphics.new_image(width, height)
    for token, bounding_box in input:
        print_token(image, token, bounding_box)
    return image


def visualize(input):
    image = create_image(input)
    cv2.imshow("image", image)
    cv2.waitKey(0)

