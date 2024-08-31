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


def poisson_editing(source, target, mask, offset=(0, 0), mixed=False):
    source = source.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0

    source_region, target_region, region_size = compute_regions(source, target, offset)
    mask = mask[source_region[0]:source_region[2], source_region[1]:source_region[3]]
    mask, mask_border = preprocess_mask(mask)
    flt_mask = mask.flatten()

    size = np.prod(region_size)
    A = scipy.sparse.identity(size, format='lil')  # Create a sparse matrix with dimensions size x size
    for i in range(region_size[0]):
        for j in range(region_size[1]):
            index = i * region_size[1] + j
            if mask[i, j] == 255:
                A[index, index] = 4
                if i == 0:
                    A[index, index] -= 1
                if i == region_size[0] - 1:
                    A[index, index] -= 1
                if j == 0:
                    A[index, index] -= 1
                if j == region_size[1] - 1:
                    A[index, index] -= 1

                if index + 1 < size and flt_mask[index + 1] == 255:
                    A[index + 1, index] = -1
                if index - 1 >= 0 and flt_mask[index - 1] == 255:
                    A[index - 1, index] = -1
                if index + mask.shape[1] < size and flt_mask[index + mask.shape[1]] == 255:
                    A[index, index + mask.shape[1]] = -1
                if index - mask.shape[1] >= 0 and flt_mask[index - mask.shape[1]] == 255:
                    A[index, index - mask.shape[1]] = -1
    A = A.tocsr()

    for channel in range(target.shape[2]):
        b = np.zeros(size, dtype=np.float32)
        t = target[target_region[0]:target_region[2], target_region[1]:target_region[3], channel]

        if len(source.shape) > 2:
            s = source[source_region[0]:source_region[2], source_region[1]:source_region[3], channel]
        else:
            s = source[source_region[0]:source_region[2], source_region[1]:source_region[3]]
        # t = t.flatten()
        # s = s.flatten()

        for i in range(region_size[0]):
            for j in range(region_size[1]):
                index = i * mask.shape[1] + j
                if mask[i, j] == 255:
                    if i > 0:
                        diff_g = s[i, j] - s[i - 1, j]
                        if mixed:
                            diff_f = t[i, j] - t[i - 1, j]
                            if abs(diff_f) > abs(diff_g):
                                b[index] += diff_f
                            else:
                                b[index] += diff_g
                        else:
                            b[index] += diff_g
                        if (i - 1, j) in mask_border:
                            b[index] += t[i - 1, j]

                    if i < mask.shape[0] - 1:
                        diff_g = s[i, j] - s[i + 1, j]
                        if mixed:
                            diff_f = t[i, j] - t[i + 1, j]
                            if abs(diff_f) > abs(diff_g):
                                b[index] += diff_f
                            else:
                                b[index] += diff_g
                        else:
                            b[index] += diff_g
                        if (i + 1, j) in mask_border:
                            b[index] += t[i + 1, j]

                    if j > 0:
                        diff_g = s[i, j] - s[i, j - 1]
                        if mixed:
                            diff_f = t[i, j] - t[i, j - 1]
                            if abs(diff_f) > abs(diff_g):
                                b[index] += diff_f
                            else:
                                b[index] += diff_g
                        else:
                            b[index] += diff_g
                        if (i, j - 1) in mask_border:
                            b[index] += t[i, j - 1]

                    if j < mask.shape[1] - 1:
                        diff_g = s[i, j] - s[i, j + 1]
                        if mixed:
                            diff_f = t[i, j] - t[i, j + 1]
                            if abs(diff_f) > abs(diff_g):
                                b[index] += diff_f
                            else:
                                b[index] += diff_g
                        else:
                            b[index] += diff_g
                        if (i, j + 1) in mask_border:
                            b[index] += t[i, j + 1]
                else:
                    if i < t.shape[0] and j < t.shape[1]:
                        b[index] = t[i, j]

        x = spsolve(A, b)
        x = x.reshape(region_size)
        x[x > 1] = 1
        x[x < 0] = 0
        target[target_region[0]:target_region[2], target_region[1]:target_region[3], channel] = x

    return target


if __name__ == '__main__':
    # source = cv2.imread('testimages/test1_src.png')
    # target = cv2.imread('testimages/test1_target.png')
    # mask = cv2.imread('testimages/test1_mask.png', cv2.IMREAD_GRAYSCALE)

    # source = cv2.imread('pera/pomodoro2.jpg', cv2.IMREAD_GRAYSCALE)
    # target = cv2.imread('pera/pera_target2.jpg')
    # mask = cv2.imread('pera/pomodoro_mask.jpg', cv2.IMREAD_GRAYSCALE)
    # source = cv2.resize(source, (500, 500))
    # target = cv2.resize(target, (500, 500))
    # mask = cv2.resize(mask, (500, 500))

    # source = cv2.imread('images1/source.jpg')
    # target = cv2.imread('images1/target.jpg')
    # mask = cv2.imread('images1/mask_sun.jpg', cv2.IMREAD_GRAYSCALE)
    # source = cv2.resize(source, (500, 500))
    # target = cv2.resize(target, (500, 500))
    # mask = cv2.resize(mask, (500, 500))

    # source = cv2.imread('images2/source.jpg')
    # target = cv2.imread('images2/target.jpg')
    # mask = cv2.imread('images2/mask.jpg', cv2.IMREAD_GRAYSCALE)
    # source = cv2.resize(source, (500, 500))
    # target = cv2.resize(target, (500, 500))
    # mask = cv2.resize(mask, (500, 500))

    # source = cv2.imread('lavagna/source2.jpg')
    # target = cv2.imread('lavagna/target.jpg')
    # mask = cv2.imread('lavagna/mask2.jpg', cv2.IMREAD_GRAYSCALE)
    # source = cv2.resize(source, (500, 500))
    # # source = 255 - source
    # # cv2.imshow('source', source)
    # target = cv2.resize(target, (500, 500))
    # mask = cv2.resize(mask, (500, 500))

    source = cv2.imread('rainbow/source.jpg')
    target = cv2.imread('rainbow/target2.jpg')
    mask = cv2.imread('rainbow/mask.jpg', cv2.IMREAD_GRAYSCALE)
    source = cv2.resize(source, (500, 500))
    target = cv2.resize(target, (500, 500))
    mask = cv2.resize(mask, (500, 500))

    # result = poisson_editing(source, target, mask, offset=(40, -30))
    # cv2.imwrite('testimages/result.png', 255 * result)
    result = poisson_editing(source, target, mask, offset=(-190, 0), mixed=True)
    cv2.imwrite('rainbow/result2_mixed_offset.jpg', 255 * result)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
