from numpy import zeros
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from numpy import cos
from numpy.core.umath import pi
from numpy.linalg import norm
from numpy import uint8
from numpy import asarray
from numpy import array
from queue import PriorityQueue
from decimal import Decimal
from decimal import getcontext
import threading
import cv2
import os
COLAB = False


"""#### Utility functions for image processing
##### The following block contains functions for :
* Image devision , Combination
* Generating basis , DCT and IDCT
* Quantization and deuantization
* zigzag conversion
* Run length encode and decode
"""


def devide_image(img):
    """Takes an image and devides it to n*m sub images... n and m are the nearest integer scale of 8 x 8
    returns the sub images (as an array), n and and m"""
    img_array = asarray(img)
    if len(img_array.shape) > 2:
        img_array = img_array[:, :, 0]
    n, m = img_array.shape
    n, m = int(n / 8), int(m / 8)
    res_devided = []
    for i in range(n):
        for j in range(m):
            res_devided.append(img_array[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8])
    return res_devided, n, m


def combine_image(blocks, n, m, limit=True):
    """Takes an array of 8x8 blocks to be combined to a nx8,mx8 image"""
    final_image = zeros((n * 8, m * 8))
    idx = 0
    for i in range(n):
        for j in range(m):
            final_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = blocks[idx]
            idx += 1
    if limit:
        return final_image.astype(uint8)
    return final_image.astype(int)

    ########################


def create_basis_mat():
    """Creating a matrix containing another basis matrices for all u and v to be used in block DCT"""
    basis_mat = np.empty(shape=(8, 8), dtype=object)
    for u in range(8):
        for v in range(8):
            # for x in range(8) :
            #     for y in range(8) :
            #         basis[x,y] = (cos((1/16)* (2*x +1)*u*pi) *  cos((1/16)* (2*y +1)*v*pi))
            """The next line is similar to the previous two for loops"""
            basis = np.fromfunction(
                lambda x, y: (
                    cos((1 / 16) * (2 * x + 1) * u * pi)
                    * cos((1 / 16) * (2 * y + 1) * v * pi)
                ),
                (8, 8),
                dtype=float,
            )
            basis_mat[u, v] = basis
    return basis_mat


def dct8_image(image_part: np.array, basis_mat: np.array = None):
    """8x8 image array --> 8x8 array of the same image after DCT
    .. the function can be provided with the basis matrix for faster execution"""
    basis = np.zeros(shape=(8, 8))
    res = np.zeros(shape=(8, 8))
    for u in range(8):
        for v in range(8):
            if (
                basis_mat is not None
            ):  # if basis matrix is provided , use it .. else create basis for each u ,v
                basis = basis_mat[u, v]
            else:
                basis = np.fromfunction(
                    lambda x, y: (
                        cos((1 / 16) * (2 * x + 1) * u * pi)
                        * cos((1 / 16) * (2 * y + 1) * v * pi)
                    ),
                    (8, 8),
                    dtype=float,
                )

            res[u, v] = np.sum(
                image_part * basis
            )  # sum of each element of basis multiplied to the corresponding element in image

    res[0, :] = res[0, :] / 2
    res[:, 0] = res[:, 0] / 2
    res[:, :] = res[:, :] / 16
    return res


def idct8_image(image_part: np.array, basis_mat: np.array = None, limit=True):
    """Similar to DCT ,8x8 DCT image array --> 8x8 array of the original image"""
    basis = np.zeros(shape=(8, 8))
    res = np.zeros(shape=(8, 8))
    for u in range(8):
        for v in range(8):
            if (
                basis_mat is not None
            ):  # if basis matrix is provided , use it .. else create basis for each u ,v
                basis = basis_mat[u, v]
            else:
                basis = np.fromfunction(
                    lambda x, y: (
                        cos((1 / 16) * (2 * x + 1) * u * pi)
                        * cos((1 / 16) * (2 * y + 1) * v * pi)
                    ),
                    (8, 8),
                    dtype=float,
                )
            res += image_part[u, v] * basis
    if limit:
        res = np.clip(res, 0, 255)
        return res.astype(uint8)
    return res.astype(int)


def create_q_table(q_type="luminance", q_factor: int = 50):
    if q_type == "luminance":
        qm = np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ]
        )
    elif q_type == "chrominance":
        qm = np.array(
            [
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ]
        )
    s = 5000 / q_factor if (q_factor < 50) else 200 - q_factor * 2
    qm_f = np.floor((qm * s - 50) / 100)
    return qm_f

    ########################


def quantize(im: np.array, qm_f):
    res = (np.round(im / qm_f)).astype(np.int32)
    return res


def dequantize(im: np.array, qm_f):
    res = (np.round(im.astype(float) * qm_f.astype(float))).astype(np.int32)
    return res


#####################


def to_zigzag(im: np.array):
    """Input 8x8 matrix --> output 64 array with zigzag pattern"""
    res = np.zeros(64, dtype=np.int32)
    x = 0
    y = 0
    dr = 1
    for i in range(32):
        res[i] = im[y, x]
        x += dr
        y -= dr
        if y == -1:
            y = 0
            dr = -1
        if x == -1:
            x = 0
            dr = 1

    x = 7
    y = 7
    dr = 1
    for i in range(32):
        res[-(i + 1)] = im[y, x]
        x -= dr
        y += dr
        if y == 8:
            y = 7
            dr = -1
        if x == 8:
            x = 7
            dr = 1

    return res


def from_zigzag(inpt: np.array):
    """Input  64 array with zigzag pattern --> output 8x8 matrix"""
    res = np.zeros((8, 8), dtype=np.int32)
    x = 0
    y = 0
    dr = 1
    for i in range(32):
        res[y, x] = inpt[i]
        x += dr
        y -= dr
        if y == -1:
            y = 0
            dr = -1
        if x == -1:
            x = 0
            dr = 1

    x = 7
    y = 7
    dr = 1
    for i in range(32):
        res[y, x] = inpt[-(i + 1)]
        x -= dr
        y += dr
        if y == 8:
            y = 7
            dr = -1
        if x == 8:
            x = 7
            dr = 1

    return res


###################
def run_len_encode(inpt: np.array):
    res = np.zeros(2 * len(inpt), dtype=np.int32)
    res_idx = 0
    idx = 0
    while idx < len(inpt):
        res[res_idx] = inpt[idx]
        res_idx += 1
        if inpt[idx] == 0:
            cnt = 1
            idx += 1
            while idx < len(inpt) and inpt[idx] == 0:
                cnt += 1
                idx += 1
            res[res_idx] = cnt
            res_idx += 1
        else:
            idx += 1
    return res[0:res_idx]


def run_len_decode(inpt: np.array):
    res = []
    i = 0
    while i < len(inpt):
        res.append(inpt[i])
        if inpt[i] == 0:
            i += 1
            for j in range(inpt[i] - 1):
                res.append(0)
        i += 1
    return np.array(res)


"""#### Huffman Encode/ Decode functions
##### The following block contains an implementation of a binary tree node , Frequency per symbol calculation function and  huffman encode/decode functions
"""


class node:
    """A class we will use to represent a symbol node in the tree"""

    left = None
    right = None
    freq = None
    symbol = None

    def __init__(self, freq=None, symbol=None, left=None, right=None) -> None:
        self.left = left
        self.right = right
        self.symbol = symbol
        self.freq = freq

    def __lt__(self, nxt):
        """Ovveriding the "Less than" class method so that the nodes can be compared by frequency and can be used easily in pq"""
        return self.freq < nxt.freq


def huffman_code(freqs: dict) -> dict:
    """Function the takes a dict of frequencies and returns  a dict of corrosponding coding"""
    result_dict = freqs.copy()
    q = PriorityQueue()
    for symbol, freq in freqs.items():
        q.put(node(symbol=symbol, freq=freq))
    node_right = None

    while q.qsize() > 1:  # q has at least 2 nodes left
        node_right = q.get()
        node_left = q.get()
        parent = node(
            freq=node_right.freq + node_left.freq,
            symbol=None,
            left=node_left,
            right=node_right,
        )
        q.put(parent)
        node_right = parent

    # traverse the tree to the coding , left --> 1 , right --> 0 , we will use recursion
    def traverse_node(nd: node, init_code: str = ""):
        if nd.left:
            traverse_node(nd.left, init_code + "1")
        if nd.right:
            traverse_node(nd.right, init_code + "0")
        if not (nd.left or nd.right):  # node for symbol
            result_dict[nd.symbol] = init_code

    traverse_node(node_right)
    return result_dict


def calculate_probs(text):
    """returns a dictionary of characters in the input text and their probabilities"""
    data_set = set(text)
    res_dict = dict()
    for i in data_set:
        res_dict[i] = 0

    for i in text:
        res_dict[i] += 1

    total = len(text)
    for key, val in res_dict.items():
        res_dict[key] = val / total

    return res_dict


def huff_encode_stream(text, code_table: dict):
    """returns a string corresponding the input text with each character mapped to a value in the code table"""
    res = ""
    for s in text:
        if s in code_table.keys():
            res += str(code_table[s])
    return res


def huff_decode_stream(text, code_table: dict):
    """Decode string/array of '1' and '0' from a code table , uses a binary tree to decode"""
    # construct tree for code table
    parent = node()
    for key, val in code_table.items():
        chld = parent
        for bt in val:
            if bt == "1":
                if not chld.left:
                    chld.left = node()
                chld = chld.left
            else:
                if not chld.right:
                    chld.right = node()
                chld = chld.right
        chld.symbol = key
    res = []

    # decode input stream using the tree
    chld: node = parent
    for ltr in text:
        if ltr == "1":
            chld = chld.left
        else:
            chld = chld.right
        if not chld.symbol == None:

            res.append(chld.symbol)
            chld = parent
    return np.array(res)


"""#### Arithmetic coding functions
##### The following block contains functions for Arithmetic encoding and decoding
* The default block length for a complete code encoding is 10 bits
* Any block length greater than 10 MAY cause cause precision problems that will result in a crash
"""

getcontext().prec = 500  # 500 precision points


def dec2bin(num, limit):  # decimal to binary string with limit
    res = ""
    for _ in range(limit):
        num *= 2
        if int(num) == 0:
            res += "0"
        else:
            res += "1"
            num -= 1

    return res


def get_symbol_range(symbol_prob: dict, code_word):
    """A functtion to get the low and high limits from a word whose symbols are contained within the dictionry
    of probabilities"""

    smbl = code_word[0]
    low = Decimal(0)
    high = Decimal(sum(symbol_prob.values()))
    for smbl in code_word:
        hght = Decimal(high - low)
        for key, val in symbol_prob.items():
            val = Decimal(val)
            if key == smbl:
                high = low + hght * val
                break
            low += hght * val

    return low, high


def encode_with_range(low, high):
    """A function to represent the limits into a binary code (as string)"""
    num = (low + high) / 2
    limit = int(np.ceil(np.log2(1 / float(high - low)) + 1))

    return dec2bin(num, limit)


def decode_bin(symbol_prob: dict, bin_word: str, n: int):
    """Decode a binary string to word of n symbols"""
    point = Decimal(0)
    for i, val in enumerate(bin_word):
        point += Decimal(int(val)) * Decimal(2 ** -(i + 1))
    low = Decimal(0)
    high = Decimal(sum(symbol_prob.values()))
    res = np.empty(n, dtype=type(list(symbol_prob.keys())[0]))

    for i in range(n):
        hght = high - low
        for key, val in symbol_prob.items():
            val = Decimal(val)
            if low + hght * val > point:
                high = low + hght * val
                res[i] = key
                break
            low += hght * val
    return res


def enocde_arithmetic(symbol_prob: dict, symbol_seq, step=10):
    """Wrapper for big code arithmetic encoding , default block size is 10"""
    code = ""
    lens = np.zeros(int(np.ceil(len(symbol_seq) / step) + 1), dtype=np.int32)
    for i in range(int(np.ceil(len(symbol_seq) / step))):
        seq = symbol_seq[i * step : (i + 1) * step]
        low, high = get_symbol_range(symbol_prob, seq)
        # print(seq)
        # print(low )
        # print(high)
        res = encode_with_range(low, high)
        code += res
        lens[i] = len(res)

    lens[-1] = len(symbol_seq)  # last element we will save the original sequence length
    return code, lens


def decode_arithmetic(symbol_prob: dict, code_seq, lens, step=10):
    """Wrapper for big code arithmetic decoding , default block size is 10"""
    result = np.zeros(step * (len(lens) - 1), dtype=np.int32)
    total = lens[-1]
    lens = lens[:-1]
    idx = 0
    for i, ln in enumerate(lens):
        # print(ln)
        seq = code_seq[idx : idx + ln]
        idx = idx + ln
        result[i * step : (i + 1) * step] = decode_bin(symbol_prob, seq, step)

    return result[:total]