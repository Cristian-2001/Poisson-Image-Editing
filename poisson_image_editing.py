import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve


def preprocess_mask(mask):
    # mask_border = np.zeros_like(mask)
    mask_border = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > 0:
                mask[i, j] = 255
            else:
                mask[i, j] = 0

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                if i > 0 and mask[i - 1, j] == 255:
                    # mask_border[i, j] = 255
                    mask_border.append((i, j))
                elif j > 0 and mask[i, j - 1] == 255:
                    # mask_border[i, j] = 255
                    mask_border.append((i, j))
                elif i < mask.shape[0] - 1 and mask[i + 1, j] == 255:
                    # mask_border[i, j] = 255
                    mask_border.append((i, j))
                elif j < mask.shape[1] - 1 and mask[i, j + 1] == 255:
                    # mask_border[i, j] = 255
                    mask_border.append((i, j))
    return mask, mask_border


# def compute_source_region(source, mask):
#     indices = np.where(mask == 255)
#     rows = indices[0]
#     cols = indices[1]
#     top, bottom = rows.min(), rows.max() + 1
#     left, right = cols.min(), cols.max() + 1
#
#     source_region = source[top:bottom, left:right]
#     region_shape = source_region.shape
#     return source_region, region_shape

def compute_regions(source, target, offset):
    source_region = (
        max(-offset[0], 0),  # right
        max(-offset[1], 0),  # top
        min(source.shape[0], target.shape[0] - offset[0]),  # left
        min(source.shape[1], target.shape[1] - offset[1])  # bottom
    )
    target_region = (
        max(offset[0], 0),  # right
        max(offset[1], 0),  # top
        min(target.shape[0], source.shape[0] + offset[0]),  # left
        min(target.shape[1], source.shape[1] + offset[1])  # bottom
    )
    region_size = (source_region[2] - source_region[0], source_region[3] - source_region[1])
    return source_region, target_region, region_size


def poisson_editing(source, target, mask, offset=(0, 0)):
    source = source.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0

    source_region, target_region, region_size = compute_regions(source, target, offset)
    mask = mask[source_region[0]:source_region[2], source_region[1]:source_region[3]]
    mask, mask_border = preprocess_mask(mask)

    # cv2.imshow("mask_border", mask_border)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    size = np.prod(region_size)
    A = scipy.sparse.identity(size, format='lil')  # Create a sparse matrix with dimensions size x size
    for i in range(region_size[0]):
        for j in range(region_size[1]):
            index = i * region_size[1] + j
            if mask[i, j] == 255:
                A[index, index] = 4

                if index + 1 < size:
                    A[index + 1, index] = -1
                if index - 1 >= 0:
                    A[index - 1, index] = -1
                if index + mask.shape[1] < size:
                    A[index, index + mask.shape[1]] = -1
                if index - mask.shape[1] >= 0:
                    A[index, index - mask.shape[1]] = -1
    A = A.tocsr()

    for channel in range(target.shape[2]):
        b = np.zeros(size, dtype=np.float32)
        t = target[target_region[0]:target_region[2], target_region[1]:target_region[3], channel]
        s = source[source_region[0]:source_region[2], source_region[1]:source_region[3], channel]
        # t = t.flatten()
        # s = s.flatten()

        for i in range(region_size[0]):
            for j in range(region_size[1]):
                counter = 0
                index = i * mask.shape[1] + j
                if mask[i, j] == 255:
                    # b[index] = 4 * s[i, j]
                    if i > 0:
                        b[index] += (s[i, j] - s[i - 1, j])
                        counter += 1
                        # if mask_border[i - 1, j] == 255:
                        if (i - 1, j) in mask_border:
                            b[index] += t[i - 1, j]
                    if i < mask.shape[0] - 1:
                        # b[index] -= s[i + 1, j]
                        b[index] += (s[i, j] - s[i + 1, j])
                        counter += 1
                        # if mask_border[i + 1, j] == 255:
                        if (i + 1, j) in mask_border:
                            b[index] += t[i + 1, j]
                    if j > 0:
                        # b[index] -= s[i, j - 1]
                        b[index] += (s[i, j] - s[i, j - 1])
                        counter += 1
                        # if mask_border[i, j - 1] == 255:
                        if (i, j - 1) in mask_border:
                            b[index] += t[i, j - 1]
                    if j < mask.shape[1] - 1:
                        # b[index] -= s[i, j + 1]
                        b[index] += (s[i, j] - s[i, j + 1])
                        counter += 1
                        # if mask_border[i, j + 1] == 255:
                        if (i, j + 1) in mask_border:
                            b[index] += t[i, j + 1]
                    # b[index] += counter * s[i, j]

                    # b[index] = 4 * s[i, j]
                    # if i > 0:
                    #     b[index] -= s[i - 1, j]
                    # if i < mask.shape[0] - 1:
                    #     b[index] -= s[i + 1, j]
                    # if j > 0:
                    #     b[index] -= s[i, j - 1]
                    # if j < mask.shape[1] - 1:
                    #     b[index] -= s[i, j + 1]
                else:
                    if i < t.shape[0] and j < t.shape[1]:
                        b[index] = t[i, j]

                    # if i == 0 or i == region_size[0] - 1 or j == 0 or j == region_size[1] - 1:
                    # if mask_border[i, j] == 255:
                    #     b[index] += t[i, j]
                    #     print("ECCOMI")
                # print(b[index])

        # b = b.astype(np.uint8)
        x = spsolve(A, b)
        x = x.reshape(region_size)
        print(x[x > 1])
        x[x > 1] = 1
        x[x < 0] = 0
        print(x[x > 1])
        # print(x)
        target[target_region[0]:target_region[2], target_region[1]:target_region[3], channel] = x
        # cv2.imshow("channel " + str(channel), s)

    return target


if __name__ == '__main__':
    source = cv2.imread('testimages/test1_src.png')
    target = cv2.imread('testimages/test1_target.png')
    mask = cv2.imread('testimages/test1_mask.png', cv2.IMREAD_GRAYSCALE)

    # source = cv2.imread('images2/source.jpg')
    # target = cv2.imread('images2/target.jpg')
    # mask = cv2.imread('images2/mask.jpg', cv2.IMREAD_GRAYSCALE)
    # source = cv2.resize(source, (500, 500))
    # target = cv2.resize(target, (500, 500))
    # mask = cv2.resize(mask, (500, 500))

    result = poisson_editing(source, target, mask, offset=(40, -30))
    cv2.imwrite('testimages/result.png', 255 * result)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
