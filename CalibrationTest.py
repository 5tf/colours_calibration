from flask import Flask, request
import numpy as np
from plantcv import plantcv as pcv
import cv2

app = Flask(__name__)


def apply_transformation_matrix(source_img, transformation_matrix):

    # split transformation_matrix
    red, green, blue, red2, green2, blue2, red3, green3, blue3 = np.split(
        transformation_matrix, 9, 1)

    # find linear, square, and cubic values of source_img color channels
    source_b, source_g, source_r = cv2.split(source_img)
    source_b2 = np.square(source_b)
    source_b3 = np.power(source_b, 3)
    source_g2 = np.square(source_g)
    source_g3 = np.power(source_g, 3)
    source_r2 = np.square(source_r)
    source_r3 = np.power(source_r, 3)

    # apply linear model to source color channels
    b = 0 + source_r * blue[0] + source_g * blue[1] + source_b * blue[
        2] + source_r2 * blue[3] + source_g2 * blue[
            4] + source_b2 * blue[5] + source_r3 * blue[6] + source_g3 * blue[
            7] + source_b3 * blue[8]
    g = 0 + source_r * green[0] + source_g * green[1] + source_b * green[
        2] + source_r2 * green[3] + source_g2 * green[
            4] + source_b2 * green[5] + source_r3 * green[6] + source_g3 * \
        green[7] + source_b3 * green[8]
    r = 0 + source_r * red[0] + source_g * red[1] + source_b * red[
        2] + source_r2 * red[3] + source_g2 * red[
            4] + source_b2 * red[5] + source_r3 * red[6] + source_g3 * red[
            7] + source_b3 * red[8]

    # merge corrected color channels onto source_image
    bgr = [b, g, r]
    corrected_img = cv2.merge(bgr)

    # round corrected_img elements to be within range and of the correct data
    # type
    corrected_img = np.rint(corrected_img)
    corrected_img[np.where(corrected_img > 255)] = 255
    corrected_img = corrected_img.astype(np.uint8)

    # return corrected_img
    return corrected_img


@app.route('/calibrate_colours', methods=['POST'])
def calibrate_colours():
    byte_img = request.data
    np_arr = np.fromstring(byte_img, dtype='uint8')
    rgb_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    df, start, space = pcv.transform.find_color_card(rgb_img=rgb_img)
    mask = pcv.transform.create_color_card_mask(rgb_img=rgb_img, radius=10,
                                                start_coord=start,
                                                spacing=space, ncols=6, nrows=4)

    headers, color_matrix = pcv.transform.get_color_matrix(rgb_img, mask)
    print(headers)
    print(color_matrix)

    xrite_values = [115., 82., 68.,
                    194., 150., 130.,
                    98., 122., 157.,
                    87., 108., 67.,
                    133., 128., 177.,
                    103., 189., 170.,
                    214., 126., 44.,
                    80., 91., 166.,
                    193., 90., 99.,
                    94., 60., 108.,
                    157., 188., 64.,
                    224., 163., 46.,
                    56., 61., 150.,
                    70., 148., 73.,
                    175., 54., 60.,
                    231., 199., 31.,
                    187., 86., 149.,
                    8., 133., 161.,
                    243., 243., 242.,
                    200., 200., 200.,
                    160., 160., 160.,
                    122., 122., 121.,
                    85., 85., 85.,
                    52., 52., 52.]

    target_color_matrix = np.zeros((len(np.unique(mask)) - 1, 4))
    row_counter = 0
    for i in np.unique(mask):
        if i != 0:
            target_color_matrix[row_counter][0] = i
            target_color_matrix[row_counter][1] = xrite_values[row_counter]
            target_color_matrix[row_counter][2] = xrite_values[row_counter + 1]
            target_color_matrix[row_counter][3] = xrite_values[row_counter + 2]
            row_counter += 1

    print(target_color_matrix)

    matrix_a, matrix_m, matrix_b = pcv.transform.get_matrix_m(
        target_color_matrix,
        color_matrix)

    print("Moore-Penrose Inverse Matrix: ")
    print(matrix_m)

    deviance, transformation_matrix = pcv.transform.calc_transformation_matrix(
        matrix_m, matrix_b)

    corrected_img = apply_transformation_matrix(rgb_img, transformation_matrix)

    pcv.print_image(corrected_img, 'corrected.jpg')


if __name__ == '__main__':
    app.run(debug=True)
