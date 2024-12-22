from PIL import Image
from IPython.display import Image as Img
import numpy as np
from numpy import zeros
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
from utils import *
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
"""#### Two wrapper functions for image encoding & decoding
##### The next two functions for image decoding / encoding utilize the previously defined functions to work on a whole image with an efficient and organized code (kinda)
"""


def encode_image(img, q_type="luminance", q_factor=50, encode_type="huff"):
    """takes input image and quantization table type ('luminance'/'chrominance' ) and value for quality factor
    returns the encoded stream, the image dimensions and the code table"""
    image_parts, n, m = devide_image(img)  # Devide image into array of arrays
    basis_mat = create_basis_mat()  # create the basis for all u,v = 0 :8

    q_table = create_q_table(q_type, q_factor)

    dct_blocks = np.array(
        [dct8_image(part, basis_mat) for part in image_parts]
    )  # apply DCT on all parts
    quantized_blocks = np.array(
        [quantize(part, q_table) for part in dct_blocks]
    )  # Quantize all parts
    zigzag_parts = np.array(
        [to_zigzag(part) for part in quantized_blocks]
    )  # Apply zigzag (2d --> 1d) for all parts
    length_coded_parts = np.array(
        [run_len_encode(part) for part in zigzag_parts], dtype=object
    )  # aplly run length encoding on all parts
    coded_combined = np.concatenate(
        length_coded_parts
    )  # Compine all parts intp one continuous array

    if encode_type == "huff":
        encode_table = huffman_code(
            freqs=calculate_probs(coded_combined)
        )  # get the code table for the frequencies of symbols in the stream
        encoded_stream = huff_encode_stream(
            text=coded_combined, code_table=encode_table
        )  # Encode the stream
    elif encode_type == "arth":
        probs_dict = calculate_probs(coded_combined)
        encoded_stream, lens = enocde_arithmetic(probs_dict, coded_combined)
        encode_table = (probs_dict, lens)

    return encoded_stream, n, m, encode_table  # return results


def decode_image(
    encoded_stream,
    n,
    m,
    encode_table,
    q_type="luminance",
    q_factor=50,
    encode_type="huff",
    limited=True,
):

    if encode_type == "huff":
        decoded_stream = huff_decode_stream(
            encoded_stream, encode_table
        )  # Decde the input stream
    elif encode_type == "arth":
        probs_dict, lens = encode_table
        decoded_stream = decode_arithmetic(probs_dict, encoded_stream, lens)

    expanded_full_length = run_len_decode(decoded_stream)  # Expand zeros in the input
    segmented_blocks_1d = np.array(
        [
            expanded_full_length[i * 64 : (i + 1) * 64]
            for i in range(int(len(expanded_full_length) / 64))
        ]
    )  # Devide the input to n*m/64 64-element arrays
    after_zigzag_parts_2d = np.array(
        [from_zigzag(seg) for seg in segmented_blocks_1d]
    )  # Apply zigzag (1d --> 2d) for all parts

    q_table = create_q_table(q_type, q_factor)

    dequantized_parts = np.array(
        [dequantize(part, q_table) for part in after_zigzag_parts_2d]
    )  # Dequantize using qiven parameters
    basis_mat = create_basis_mat()  # create the basis for all u,v = 0 :8
    idct_blocks = np.array(
        [idct8_image(part, basis_mat, limit=limited) for part in dequantized_parts]
    )  # apply IDCT on all parts
    final_image = combine_image(
        blocks=idct_blocks, n=n, m=m, limit=limited
    )  # Compine all parts to one single image
    return final_image  # return full image


"""# Lossless image encoding decoding"""


def encode_image_lossless(img):
    """Lossless image encoding using delta encoding and Huffman coding"""
    # Convert image to numpy array
    img_array = np.array(img)
    if len(img_array.shape) > 2:
        img_array = img_array[:, :, 0]  # Convert to grayscale if needed

    # Store dimensions
    height, width = img_array.shape

    # Store first column for reconstruction
    first_column = img_array[:, 0].copy()

    # Delta encoding (difference between adjacent pixels)
    delta_encoded = np.zeros_like(img_array)
    delta_encoded[:, 0] = img_array[:, 0]
    delta_encoded[:, 1:] = img_array[:, 1:] - img_array[:, :-1]

    # Flatten the delta encoded array
    flattened = delta_encoded.flatten()

    # Huffman encoding
    huffman_table = huffman_code(calculate_probs(flattened))
    encoded_stream = huff_encode_stream(flattened, huffman_table)

    return {
        "stream": encoded_stream,
        "height": height,
        "width": width,
        "huffman_table": huffman_table,
        "first_column": first_column,
    }


def decode_image_lossless(encoded_data):
    """Lossless image decoding"""
    # Extract data from encoded dictionary
    height = encoded_data["height"]
    width = encoded_data["width"]
    huffman_table = encoded_data["huffman_table"]
    first_column = encoded_data["first_column"]

    # Huffman decode
    decoded_data = huff_decode_stream(encoded_data["stream"], huffman_table)

    # Reshape to original dimensions
    delta_decoded = decoded_data.reshape(height, width)

    # Reconstruct original image from deltas
    reconstructed = np.zeros_like(delta_decoded)
    reconstructed[:, 0] = delta_decoded[:, 0]  # First column remains unchanged

    # Cumulative sum along rows to recover original values
    for i in range(1, width):
        reconstructed[:, i] = reconstructed[:, i - 1] + delta_decoded[:, i]

    return reconstructed



"""#### Motion vectors getting and compensating functions
* The motion vector function checks the motion in 9 possibilities (same position , up , down , left , right , up left , up right , down left , down right)
* Best vector is choses by the least square error
* For each image , a matrix of vectors is generated for each block in the image
"""


def get_motion_mat(ref_frame: np.array, next_frame: np.array):
    """Returns a matrix of vectors for each 8x8 block in the image"""
    n, m = ref_frame.shape
    n, m = int(n / 8), int(m / 8)
    motion_mat = np.empty(dtype=object, shape=(n, m))
    for i in range(n):
        for j in range(m):
            min_err = norm(
                ref_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
            )
            motion_mat[i, j] = (0, 0)  # no motion
            if i > 0:
                err = norm(
                    ref_frame[(i - 1) * 8 : (i) * 8, j * 8 : (j + 1) * 8]
                    - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                )  # up block
                if err < min_err:
                    min_err = err
                    motion_mat[i, j] = (-1, 0)  # up

            if i > 0 and j > 0:
                err = norm(
                    ref_frame[(i - 1) * 8 : (i) * 8, (j - 1) * 8 : (j) * 8]
                    - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                )  # up left block
                if err < min_err:
                    min_err = err
                    motion_mat[i, j] = (-1, -1)  # up left

            if i > 0 and j < m - 1:
                err = norm(
                    ref_frame[(i - 1) * 8 : (i) * 8, (j + 1) * 8 : (j + 2) * 8]
                    - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                )  # up right block
                if err < min_err:
                    min_err = err
                    motion_mat[i, j] = (-1, 1)  # up right

            if i < n - 1:
                err = norm(
                    ref_frame[(i + 1) * 8 : (i + 2) * 8, (j) * 8 : (j + 1) * 8]
                    - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                )  # down block
                if err < min_err:
                    min_err = err
                    motion_mat[i, j] = (1, 0)  # down

            if i < n - 1 and j > 0:
                err = norm(
                    ref_frame[(i + 1) * 8 : (i + 2) * 8, (j - 1) * 8 : (j) * 8]
                    - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                )  # down left block
                if err < min_err:
                    min_err = err
                    motion_mat[i, j] = (1, -1)  # down left

            if i < n - 1 and j < m - 1:
                err = norm(
                    ref_frame[(i + 1) * 8 : (i + 2) * 8, (j + 1) * 8 : (j + 2) * 8]
                    - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                )  # down right block
                if err < min_err:
                    min_err = err
                    motion_mat[i, j] = (1, 1)  # down right

            if j > 0:
                err = norm(
                    ref_frame[(i) * 8 : (i + 1) * 8, (j - 1) * 8 : (j) * 8]
                    - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                )  # left block
                if err < min_err:
                    min_err = err
                    motion_mat[i, j] = (0, -1)  # left

            if j < m - 1:
                err = norm(
                    ref_frame[(i) * 8 : (i + 1) * 8, (j + 1) * 8 : (j + 2) * 8]
                    - next_frame[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                )  # right block
                if err < min_err:
                    min_err = err
                    motion_mat[i, j] = (0, 1)  # right

    return motion_mat


def apply_motion(ref_frame, motion_mat):
    """
    Constructs a new frame using the refrence frame and the motion vectors matrix
    """
    n, m = ref_frame.shape
    n, m = int(n / 8), int(m / 8)
    next_constructed = np.zeros(shape=(n * 8, m * 8), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            ii = i + motion_mat[i, j][0]  # y axis
            jj = j + motion_mat[i, j][1]  # x axis
            next_constructed[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = ref_frame[
                ii * 8 : (ii + 1) * 8, jj * 8 : (jj + 1) * 8
            ]

    return next_constructed


"""#### Video encoding function
* Uses the CV2 video library to read /write video
* The idea of operation is to read the frame , constructs the encoded version and rewrites it to an output file
* The functions stores only two frames in the memory (current and refrence)
* Refrence frame is encoded version of the previous current frame and is updated every frame
"""


def encode_vid(
    video_path,
    out_name,
    runtime_secs=2,
    rgb=True,
    encode_type="huff",
    display_frames=True,
    q_factor=50,
):

    # Init video capture and write classes
    cv2.destroyAllWindows()
    vid_cap = cv2.VideoCapture(video_path)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    length = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    _, first_frame = vid_cap.read()
    n, m, _ = first_frame.shape
    n, m = int(n / 8), int(m / 8)

    total_expected_length_bits = n * m * 64 * runtime_secs * fps * 3 * 8

    expected_coded_length = 0

    # Init the refrence frame by encoding and decoding the first frame
    ref_frame = np.zeros(shape=(n * 8, m * 8, 3), dtype=np.uint8)
    for i in range(3):
        encoded_stream, n, m, huff_table = encode_image(
            first_frame[:, :, i], q_factor=q_factor, encode_type=encode_type
        )
        expected_coded_length += len(encoded_stream)
        ref_frame[:, :, i] = decode_image(
            encoded_stream, n, m, huff_table, q_factor=q_factor, encode_type=encode_type
        )

    writer = cv2.VideoWriter(
        out_name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (m * 8, n * 8)
    )

    # save first frame  to disk
    writer.write(ref_frame)
    f0 = 0
    # frame counter

    while vid_cap.isOpened() and f0 <= fps * runtime_secs:
        # Capture frame-by-frame
        print(f"Frame {f0} out of {int(fps * runtime_secs)}")
        ret, frame = vid_cap.read()
        if ret == True:  # If faliure return
            frame = frame[0 : n * 8, 0 : m * 8, :]  # crop frame to multiples of 8
            f0 += 1
            constructed_frame = np.zeros(
                shape=(n * 8, m * 8, 3), dtype=np.uint8
            )  # Buffer matrix to store the constructed frame
            motion_vects = get_motion_mat(
                ref_frame=ref_frame[:, :, 0], next_frame=frame[:, :, 0]
            )  # get motion vectors with respect to channel 0
            code_len_buffer = np.zeros(3)

            # Encoding the motion vectors
            motion_vects_flattened = motion_vects.flatten()  # covert to 1d

            h_table = huffman_code(calculate_probs(motion_vects_flattened))
            encoded_stream = huff_encode_stream(
                motion_vects_flattened, h_table
            )  # encoding the motion vectors

            expected_coded_length += len(
                encoded_stream
            )  # add the encoded motion vectors length
            decoded_motion = huff_decode_stream(
                encoded_stream, code_table=h_table
            )  # decode motion vectors

            # reshape to 2d
            motion_vects = np.empty(dtype=object, shape=(n, m))
            idx = 0
            for i in range(n):
                for j in range(m):
                    motion_vects[i, j] = decoded_motion[idx]
                    idx += 1

            def encode_channel(i):  # encode color channel
                # get moved predicted frame using the motion vectors and refrence frame
                moved_frame = apply_motion(
                    motion_mat=motion_vects, ref_frame=ref_frame[:, :, i]
                )
                # Calculate residuals
                resids = -moved_frame.astype(np.int32) + frame[:, :, i].astype(np.int32)

                # Encode residuals
                encoded_stream, n, m, huff_table = encode_image(
                    resids, q_factor=q_factor, encode_type=encode_type
                )

                # Measure the coded stream length
                code_len_buffer[i] = len(encoded_stream)

                # Decode to get the residuals back
                decoded_resids = decode_image(
                    encoded_stream,
                    n,
                    m,
                    huff_table,
                    limited=False,
                    q_factor=q_factor,
                    encode_type=encode_type,
                )

                # construct current frame
                constructed_frame[:, :, i] = (
                    np.clip(
                        moved_frame.astype(np.int32) + decoded_resids.astype(np.int32),
                        0,
                        255,
                    )
                ).astype(np.uint8)

            # Each colr channel will be handeled by a thread
            t1 = threading.Thread(target=encode_channel, args=(0,))
            t2 = threading.Thread(target=encode_channel, args=(1,))
            t3 = threading.Thread(target=encode_channel, args=(2,))

            t1.start()
            t2.start()
            t3.start()

            t1.join()
            t2.join()
            t3.join()

            # Update expected code Length ( sum of the length of each color channel stream)
            expected_coded_length += np.sum(code_len_buffer)
            # Update refrence frame
            ref_frame = constructed_frame
            # Save frame to disk
            writer.write(constructed_frame)

            if display_frames:
                if not COLAB:  # cv2.imshow doesnt run on google colab
                    cv2.imshow("Frame", constructed_frame)
                else:
                    cv2_imshow(constructed_frame)
                cv2.waitKey(20)

        # Break the loop if frame read faliure
        else:
            break

    # Release resources
    vid_cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Print results
    print("Quantization table used :")
    print(create_q_table(q_factor=q_factor))
    print("Done")
    print(f"Total expected Length {total_expected_length_bits} bit")
    print(f"Expected encoded length {expected_coded_length} bit")
    print(
        f"Compression ratio  {total_expected_length_bits/expected_coded_length :.2f} : 1"
    )


"""# encode video lossless"""


def encode_vid_lossless(video_path, out_name, runtime_secs=2, display_frames=True):
    """Lossless video encoding using frame-by-frame compression"""

    # Init video capture and write classes
    cv2.destroyAllWindows()
    vid_cap = cv2.VideoCapture(video_path)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    # Read first frame
    _, first_frame = vid_cap.read()
    n, m, _ = first_frame.shape
    n, m = int(n / 8), int(m / 8)

    total_original_size = 0
    total_compressed_size = 0

    # Init the reference frame by encoding and decoding the first frame
    ref_frame = np.zeros(shape=(n * 8, m * 8, 3), dtype=np.uint8)
    for i in range(3):
        # Calculate original size
        total_original_size += first_frame[:, :, i].nbytes

        # Encode channel using lossless encoding
        encoded_data = encode_image_lossless(Image.fromarray(first_frame[:, :, i]))
        total_compressed_size += len(encoded_data["stream"]) / 8

        # Decode channel
        ref_frame[:, :, i] = decode_image_lossless(encoded_data)

    writer = cv2.VideoWriter(
        out_name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (m * 8, n * 8)
    )

    # Save first frame to disk
    writer.write(ref_frame)
    f0 = 0  # frame counter

    while vid_cap.isOpened() and f0 <= fps * runtime_secs:
        print(f"Frame {f0} out of {int(fps * runtime_secs)}")
        ret, frame = vid_cap.read()

        if ret == True:
            frame = frame[0 : n * 8, 0 : m * 8, :]  # crop frame to multiples of 8
            f0 += 1
            constructed_frame = np.zeros(shape=(n * 8, m * 8, 3), dtype=np.uint8)

            def encode_channel(i):
                # Calculate original size

                # Encode channel using lossless encoding
                encoded_data = encode_image_lossless(Image.fromarray(frame[:, :, i]))

                # Decode channel
                constructed_frame[:, :, i] = decode_image_lossless(encoded_data)

            # Each color channel will be handled by a thread
            t1 = threading.Thread(target=encode_channel, args=(0,))
            t2 = threading.Thread(target=encode_channel, args=(1,))
            t3 = threading.Thread(target=encode_channel, args=(2,))

            t1.start()
            t2.start()
            t3.start()
            t1.join()
            t2.join()
            t3.join()

            # Update reference frame
            ref_frame = constructed_frame
            # Save frame to disk
            writer.write(constructed_frame)

            if display_frames:
                if not COLAB:  # cv2.imshow doesn't run on google colab
                    cv2.imshow("Frame", constructed_frame)
                else:
                    cv2_imshow(constructed_frame)
                cv2.waitKey(20)
        else:
            break

    # Release resources
    vid_cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Print results
    print("Done")
    print(f"Total original size: {total_original_size/1024/1024:.2f} MB")
    print(f"Total compressed size: {total_compressed_size/1024/1024:.2f} MB")
    print(f"Compression ratio: {total_original_size/total_compressed_size:.2f}:1")


# Test image encoding decoding



def compare_compression_methods(image_path):
    # Load image
    original_img = Image.open(image_path).convert("L")  # Convert to grayscale
    original_array = np.array(original_img)

    # Calculate dimensions that are multiples of 8
    height, width = original_array.shape
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8

    # Crop to multiple of 8
    original_array = original_array[:new_height, :new_width]

    print(f"Original dimensions: {height}x{width}")
    print(f"Adjusted dimensions: {new_height}x{new_width}")

    # 1. Lossy Compression
    encoded_stream_lossy, n, m, encode_table = encode_image(
        img=original_array, q_type="luminance", q_factor=50, encode_type="huff"
    )

    decoded_img_lossy = decode_image(
        encoded_stream=encoded_stream_lossy,
        n=n,
        m=m,
        encode_table=encode_table,
        q_type="luminance",
        q_factor=50,
        encode_type="huff",
        limited=True,
    )

    # 2. Lossless Compression
    encoded_data = encode_image_lossless(original_array)  # Use cropped array
    decoded_img_lossless = decode_image_lossless(encoded_data)

    # Calculate compression ratios
    original_size = original_array.nbytes
    lossy_size = len(encoded_stream_lossy) / 8
    lossless_size = len(encoded_data["stream"]) / 8

    lossy_ratio = original_size / lossy_size
    lossless_ratio = original_size / lossless_size

    # Calculate PSNR for lossy compression
    mse = np.mean((original_array - decoded_img_lossy) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float("inf")

    # Verify lossless compression
    is_lossless = np.array_equal(original_array, decoded_img_lossless)

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"Original Image\n{original_array.shape}")
    plt.imshow(original_array, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Lossy Compression\nRatio: {lossy_ratio:.2f}:1\nPSNR: {psnr:.2f}dB")
    plt.imshow(decoded_img_lossy, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(
        f"Lossless Compression\nRatio: {lossless_ratio:.2f}:1\nPerfect: {is_lossless}"
    )
    plt.imshow(decoded_img_lossless, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    # Save the figure before showing it
    plt.savefig("compression_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print detailed results
    print(f"\nOriginal size: {original_size} bytes")
    print(f"Lossy compression ratio: {lossy_ratio:.2f}:1 (PSNR: {psnr:.2f}dB)")
    print(f"Lossless compression ratio: {lossless_ratio:.2f}:1")
    print(f"Lossless verification: {is_lossless}")


# compare_compression_methods("./multimediaProject/imageTest.jpg")
from PIL import Image, ImageDraw, ImageFont


def compare_compression_methods_rgb(image_path):
    # Load image (keeping RGB channels)
    original_img = Image.open(image_path)
    original_array = np.array(original_img)

    # Calculate dimensions that are multiples of 8
    height, width = original_array.shape[:2]
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    original_array = original_array[:new_height, :new_width]

    # Initialize arrays for storing encoded data
    encoded_lossy_data = []
    encoded_lossless_data = []

    # Process and save each channel separately
    for channel in range(3):
        # Lossy compression
        encoded_stream_lossy, n, m, encode_table = encode_image(
            img=original_array[:, :, channel],
            q_type="luminance",
            q_factor=50,
            encode_type="huff",
        )
        encoded_lossy_data.append({
            'stream': np.array(list(encoded_stream_lossy)),
            'n': n,
            'm': m,
            'table': encode_table
        })

        # Lossless compression
        encoded_lossless = encode_image_lossless(original_array[:, :, channel])
        encoded_lossless_data.append(encoded_lossless)

    # Save encoded data
    np.savez('encoded_lossy.npz', 
        data=encoded_lossy_data,
        shape=(new_height, new_width)
    )
    np.savez('encoded_lossless.npz', 
        data=encoded_lossless_data,
        shape=(new_height, new_width)
    )

    # Load and decode data
    decoded_img_lossy = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    decoded_img_lossless = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    with np.load('encoded_lossy.npz', allow_pickle=True) as data:
        encoded_lossy_data = data['data']
        for channel in range(3):
            channel_data = encoded_lossy_data[channel]
            encoded_stream = "".join(channel_data['stream'].astype(str))
            decoded_img_lossy[:, :, channel] = decode_image(
                encoded_stream,
                channel_data['n'],
                channel_data['m'],
                channel_data['table'],
                q_type="luminance",
                q_factor=50,
                encode_type="huff",
                limited=True,
            )

    with np.load('encoded_lossless.npz', allow_pickle=True) as data:
        encoded_lossless_data = data['data']
        for channel in range(3):
            decoded_img_lossless[:, :, channel] = decode_image_lossless(encoded_lossless_data[channel])

    # Calculate metrics
    original_size = original_array.nbytes
    lossy_size = os.path.getsize('encoded_lossy.npz')
    lossless_size = os.path.getsize('encoded_lossless.npz')
    
    lossy_ratio = original_size / lossy_size
    lossless_ratio = original_size / lossless_size

    # Calculate PSNR and SSIM
    psnr_lossy = peak_signal_noise_ratio(original_array, decoded_img_lossy)
    psnr_lossless = peak_signal_noise_ratio(original_array, decoded_img_lossless)

    original_gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
    lossy_gray = cv2.cvtColor(decoded_img_lossy, cv2.COLOR_RGB2GRAY)
    lossless_gray = cv2.cvtColor(decoded_img_lossless, cv2.COLOR_RGB2GRAY)

    ssim_lossy, _ = structural_similarity(original_gray, lossy_gray, full=True)
    ssim_lossless, _ = structural_similarity(original_gray, lossless_gray, full=True)

    # Create figure with results
    plt.figure(figsize=(15, 5))

    # Add detailed information to titles
    original_title = (
        f"Original Image\n"
        f"Size: {original_size/1024:.1f} KB\n"
        f"Dimensions: {original_array.shape}"
    )

    lossy_title = (
        f"Lossy Compression\n"
        f"Size: {lossy_size/1024:.1f} KB\n"
        f"Ratio: {lossy_ratio:.1f}:1\n"
        f"PSNR: {psnr_lossy:.2f} dB\n"  # Add PSNR
        f"SSIM: {ssim_lossy:.4f}"  # Add SSIM
    )
    # Verify lossless compression
    is_lossless = np.array_equal(original_array, decoded_img_lossless)
    lossless_title = (
        f"Lossless Compression\n"
        f"Size: {lossless_size/1024:.1f} KB\n"
        f"Ratio: {lossless_ratio:.1f}\n"
        f"Perfect Recovery: {is_lossless}\n"
        f"PSNR: {psnr_lossless:.2f} dB\n"  # Add PSNR (should be inf for lossless)
        f"SSIM: {ssim_lossless:.4f}"  # Add SSIM (should be 1.0 for lossless)
    )

    plt.subplot(1, 3, 1)
    plt.title(original_title)
    plt.imshow(original_array)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(lossy_title)
    plt.imshow(decoded_img_lossy)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(lossless_title)
    plt.imshow(decoded_img_lossless)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(image_path+"_compression_comparison.png", dpi=1200, bbox_inches="tight")
    plt.show()

    # Print detailed results
    print(f"\nDetailed Results:")
    print(f"Original size: {original_size/1024:.1f} KB")
    print(f"Lossy compressed size: {lossy_size/1024:.1f} KB")
    print(f"Lossy compression ratio: {lossy_ratio:.1f}:1")
    print(f"Lossy PSNR: {psnr_lossy:.2f} dB")  # Print PSNR
    print(f"Lossy SSIM: {ssim_lossy:.4f}")  # Print SSIM
    print(f"Lossless compressed size: {lossless_size/1024:.1f} KB")
    print(f"Lossless compression ratio: {lossless_ratio:.1f}:1")
    print(f"Lossless PSNR: {psnr_lossless:.2f} dB")  # Print PSNR (should be inf)
    print(f"Lossless SSIM: {ssim_lossless:.4f}")  # Print SSIM (should be 1.0)
    print(f"Lossless verification: {is_lossless}")
  

### image test  
compare_compression_methods_rgb("./imageResults/imageTest.jpg")
compare_compression_methods_rgb("./imageResults/image6.jpeg")

# """# video test"""
# encode_vid_lossless("./vidTest.mp4", "./losslessVid.avi", runtime_secs=5, display_frames=False)
# video_path = "./vidTest.mp4"
# encode_vid(video_path, "lossyVid.avi", 5, encode_type="huff")
