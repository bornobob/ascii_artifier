import skimage.io as io
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import math
from multiprocessing import Pool
import time

THREADS = 4
OUTFILE = 'outfile.txt'
kernel_definitions = {
    '|': './kernels/1.png',
    '/': './kernels/2.png',
    '\\': './kernels/3.png',
    '_': './kernels/4.png',
    '-': './kernels/5.png',
    '=': './kernels/6.png',
    ' ': './kernels/7.png'
}


def kernel_from_file(filename):
    img = io.imread(filename, as_gray=True).astype(np.int)
    img = np.invert(img)
    return (img + 2) * 3 - 1


def load_kernels(kernel_files):
    kernels = {}
    for k, v in kernel_files.items():
        kernels[k] = kernel_from_file(v)
        if k == ' ':
            kernels[k] += 1
    return kernels


def ascii_artify_partial_image(image, y_num_from, y_num_until, kernels, pih, piw):
    result = ''
    for r in range(y_num_from, y_num_until):
        for c in range(math.floor(len(image[r]) / piw)):
            image_part = [
                [image[y][x][0] / 255 for x in range(c * piw, (c + 1) * piw)]
                for y in range(r * pih, (r + 1) * pih)
            ]
            result += find_closest_char(image_part, kernels)
        result = result.rstrip() + '\n'
    return result


def find_closest_char(partial_image, kernels):
    partial_image = np.asarray(partial_image)
    best, best_char = -999, None
    for k, v in kernels.items():
        print(v)
        print(partial_image)
        total = np.sum(v * partial_image)
        if total > best:
            best, best_char = total, k
    return best_char


def ascii_artify_multi_process(image, kernels, pih, piw):
    y_len = len(image)
    y_num = math.floor(y_len / pih)
    with Pool(THREADS) as pool:
        lines_per_thread = [(y_num // THREADS)] * THREADS
        lines_per_thread[0] += y_num - (y_num // THREADS) * THREADS
        running_lines = 0
        workers = []
        for i in lines_per_thread:
            workers.append((image, running_lines, running_lines + i, kernels, pih, piw))
            running_lines += i
        result = pool.starmap(ascii_artify_partial_image, workers)
        return ''.join(result)


if __name__ == '__main__':
    start = time.time()
    try:
        img = np.array(Image.open(IMAGE_PATH).convert('LA'))
        kernel_data = load_kernels(kernel_definitions)
        partial_image_width, partial_image_height = kernel_data[' '].shape
        if THREADS > 1:
            res = ascii_artify_multi_process(img, kernel_data, partial_image_height, partial_image_width)
        else:
            res = ascii_artify_partial_image(img, 0, math.floor(len(img) / partial_image_height), kernel_data,
                                             partial_image_height, partial_image_width)
        with open(OUTFILE, 'w') as f:
            f.write(res)
    except FileNotFoundError:
        print('The file given could not be found')
    finally:
        print('Done, took {:.2f} seconds'.format(time.time() - start))
