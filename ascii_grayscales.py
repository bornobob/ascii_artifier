from PIL import ImageFont, Image, ImageDraw
import numpy as np
import math
from multiprocessing import Pool
import time


X_CONST, Y_CONST = 1, 2
CHARACTERS = '!"#$%&\'()*+,-./:;?@[\\]^`{|}~_ '
THREADS = 4
IMAGE_PATH = 'images/mario.jpg'
OUTFILE = 'outfile.txt'
MULTITHREAD = THREADS > 1


def ascii_artify_partial_image(image, y_num_from, y_num_until, letters_val):
    result = ''
    for r in range(y_num_from, y_num_until):
        for c in range(math.floor(len(image[r]) / X_CONST)):
            image_part = [
                [image[y][x][0] / 255 for x in range(c * X_CONST, (c + 1) * X_CONST)]
                for y in range(r * Y_CONST, (r + 1) * Y_CONST)
            ]
            result += find_closest_char(np.mean(image_part), letters_val)
        result = result.rstrip() + '\n'
    return result


def find_closest_char(value, letter_values):
    index = (np.abs(letter_values - value)).argmin()
    return CHARACTERS[index]


def generate_letters_val():
    letters_val = np.zeros(len(CHARACTERS))
    font = ImageFont.truetype(font='cournew.TTF', size=22)
    for idx, x in enumerate(CHARACTERS):
        img = Image.new('RGBA', (14, 21), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, -4), x, (0, 0, 0), font=font)
        ImageDraw.Draw(img)
        image = np.array(img)
        img.save('letters/{}.png'.format(idx))
        letters_val[idx] = 10 * (1 - (np.mean(image) / 255))
    return letters_val


def ascii_artify_multi_process(image, letters_val):
    y_len = len(image)
    y_num = math.floor(y_len / Y_CONST)
    with Pool(THREADS) as pool:
        lines_per_thread = [(y_num // THREADS)] * THREADS
        lines_per_thread[0] += y_num - (y_num // THREADS) * THREADS
        running_lines = 0
        workers = []
        for i in lines_per_thread:
            workers.append((image, running_lines, running_lines + i, letters_val))
            running_lines += i
        result = pool.starmap(ascii_artify_partial_image, workers)
        return ''.join(result)


if __name__ == '__main__':
    letters_val = generate_letters_val()
    start = time.time()
    try:
        image = np.array(Image.open(IMAGE_PATH).convert('LA'))
        if MULTITHREAD:
            result = ascii_artify_multi_process(image, letters_val)
        else:
            result = ascii_artify_partial_image(image, 0, math.floor(len(image) / Y_CONST), letters_val)
        with open(OUTFILE, 'w') as f:
            f.write(result)
    except FileNotFoundError:
        print('The file given could not be found')
    finally:
        print('Done, took {:.2f} seconds'.format(time.time() - start))
