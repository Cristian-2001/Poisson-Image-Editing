import numpy as np
# import cv2
from PIL import Image, ImageOps
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


def poisson_editing(source, target, mask, offset=(0, 0), mixing=False):
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

    if len(target.shape) < 3:
        target = np.repeat(target[:, :, np.newaxis], 3, axis=2)
    # print(target.shape)

    for channel in range(target.shape[2]):
        b = np.zeros(size, dtype=np.float32)

        if len(target.shape) > 2:
            t = target[target_region[0]:target_region[2], target_region[1]:target_region[3], channel]
        else:
            t = target[target_region[0]:target_region[2], target_region[1]:target_region[3]]

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
                        if mixing:
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
                        if mixing:
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
                        if mixing:
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
                        if mixing:
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

        if len(target.shape) > 2:
            target[target_region[0]:target_region[2], target_region[1]:target_region[3], channel] = x
        else:
            target[target_region[0]:target_region[2], target_region[1]:target_region[3]] = x

    return target


def texture_flattening(source, mask):
    source = source.astype(np.float32) / 255.0

    tmp_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    tmp_source = np.uint8(tmp_source * 255)
    edges = cv2.Canny(tmp_source, 100, 200)

    mask, mask_border = preprocess_mask(mask)
    flt_mask = mask.flatten()

    size = np.prod(source.shape[:2])
    A = scipy.sparse.identity(size, format='lil')
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            index = i * source.shape[1] + j
            if mask[i, j] == 255:
                A[index, index] = 4
                if i == 0:
                    A[index, index] -= 1
                if i == source.shape[0] - 1:
                    A[index, index] -= 1
                if j == 0:
                    A[index, index] -= 1
                if j == source.shape[1] - 1:
                    A[index, index] -= 1

                if index + 1 < size and flt_mask[index + 1] == 255:
                    A[index + 1, index] = -1
                if index - 1 >= 0 and flt_mask[index - 1] == 255:
                    A[index - 1, index] = -1
                if index + source.shape[1] < size and flt_mask[index + source.shape[1]] == 255:
                    A[index, index + source.shape[1]] = -1
                if index - source.shape[1] >= 0 and flt_mask[index - source.shape[1]] == 255:
                    A[index, index - source.shape[1]] = -1
    A = A.tocsr()

    for channel in range(source.shape[2]):
        b = np.zeros(size, dtype=np.float32)
        s = source[:, :, channel]

        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                index = i * source.shape[1] + j
                if mask[i, j] == 255:
                    if i > 0:
                        diff_g = s[i, j] - s[i - 1, j]
                        if edges[i, j] > 0:
                            b[index] += diff_g
                        else:
                            b[index] += 0
                        if (i - 1, j) in mask_border:
                            b[index] += s[i - 1, j]

                    if i < mask.shape[0] - 1:
                        diff_g = s[i, j] - s[i + 1, j]
                        if edges[i, j] > 0:
                            b[index] += diff_g
                        else:
                            b[index] += 0
                        if (i + 1, j) in mask_border:
                            b[index] += s[i + 1, j]

                    if j > 0:
                        diff_g = s[i, j] - s[i, j - 1]
                        if edges[i, j] > 0:
                            b[index] += diff_g
                        else:
                            b[index] += 0
                        if (i, j - 1) in mask_border:
                            b[index] += s[i, j - 1]

                    if j < mask.shape[1] - 1:
                        diff_g = s[i, j] - s[i, j + 1]
                        if edges[i, j] > 0:
                            b[index] += diff_g
                        else:
                            b[index] += 0
                        if (i, j + 1) in mask_border:
                            b[index] += s[i, j + 1]
                else:
                    b[index] = s[i, j]

            x = spsolve(A, b)
            x = x.reshape(source.shape[:2])
            x[x > 1] = 1
            x[x < 0] = 0
            source[:, :, channel] = x

    return source


def save_images(dir, source, target, mask):
    source.save(dir + '/risultati_finali/source_finale.jpg', 'JPEG')
    target.save(dir + '/risultati_finali/target_finale.jpg', 'JPEG')
    mask.save(dir + '/risultati_finali/mask_finale.jpg', 'JPEG')


if __name__ == '__main__':
    # source = Image.open('testimages/test1_src.png')
    # target = Image.open('testimages/test1_target.png')
    # mask = Image.open('testimages/test1_mask.png')
    # mask = ImageOps.grayscale(mask)
    #
    # source_array = np.array(source)
    # target_array = np.array(target)
    # mask_array = np.array(mask)
    #
    # result = poisson_editing(source_array, target_array, mask_array, offset=(40, -30))
    # im_result = Image.fromarray((result * 255).astype(np.uint8))
    # im_result.save('testimages/result.png', 'PNG')
    # im_result.show()

    '''Pera'''
    # source = Image.open('pera/pomodoro2.jpg')
    # source = ImageOps.grayscale(source)
    # target = Image.open('pera/pera_target2.jpg')
    # mask = Image.open('pera/pomodoro_mask.jpg')
    # mask = ImageOps.grayscale(mask)
    #
    # source = source.resize((source.size[0] // 4, source.size[1] // 4))
    # target = target.resize((target.size[0] // 4, target.size[1] // 4))
    # mask = mask.resize((mask.size[0] // 4, mask.size[1] // 4))
    # save_images("pera", source, target, mask)
    #
    # source_array = np.array(source)
    # target_array = np.array(target)
    # mask_array = np.array(mask)
    #
    # result = poisson_editing(source_array, target_array, mask_array, offset=(50, 30), mixing=True)
    # im_result = Image.fromarray((result * 255).astype(np.uint8))
    # im_result.save('pera/risultati_finali/result_monochrome_mixed.jpg', 'JPEG')
    # im_result.show()

    '''Lavagna'''
    # source = Image.open('lavagna/source2.jpg')
    # target = Image.open('lavagna/target.jpg')
    # mask = Image.open('lavagna/mask2.jpg')
    # mask = ImageOps.grayscale(mask)
    #
    # source = source.resize((source.size[0] // 4, source.size[1] // 4))
    # target = target.resize((target.size[0] // 4, target.size[1] // 4))
    # mask = mask.resize((mask.size[0] // 4, mask.size[1] // 4))
    # save_images("lavagna", source, target, mask)
    #
    # source_array = np.array(source)
    # target_array = np.array(target)
    # mask_array = np.array(mask)
    #
    # result = poisson_editing(source_array, target_array, mask_array, mixing=True)
    # im_result = Image.fromarray((result * 255).astype(np.uint8))
    # im_result.save('lavagna/risultati_finali/result_mixed.jpg', 'JPEG')
    # im_result.show()

    '''Sea'''
    # source = Image.open('sea/source.jpg')
    # target = Image.open('sea/target.jpg')
    # mask = Image.open('sea/mask.jpg')
    # mask = ImageOps.grayscale(mask)
    #
    # source = source.resize((source.size[0] // 4, source.size[1] // 4))
    # target = target.resize((target.size[0] // 4, target.size[1] // 4))
    # mask = mask.resize((mask.size[0] // 4, mask.size[1] // 4))
    # save_images("sea", source, target, mask)
    #
    # source_array = np.array(source)
    # target_array = np.array(target)
    # mask_array = np.array(mask)
    #
    # result = poisson_editing(source_array, target_array, mask_array, offset=(100, -100), mixing=False)
    # im_result = Image.fromarray((result * 255).astype(np.uint8))
    # im_result.save('sea/risultati_finali/result.jpg', 'JPEG')
    # im_result.show()

    '''Rainbow'''
    # source = Image.open('rainbow/source.jpg')
    # target = Image.open('rainbow/target2.jpg')
    # mask = Image.open('rainbow/mask.jpg')
    # mask = ImageOps.grayscale(mask)
    #
    # source = source.resize((source.size[0] // 4, source.size[1] // 4))
    # target = target.resize((target.size[0] // 4, target.size[1] // 4))
    # mask = mask.resize((mask.size[0] // 4, mask.size[1] // 4))
    # save_images("rainbow", source, target, mask)
    #
    # source_array = np.array(source)
    # target_array = np.array(target)
    # mask_array = np.array(mask)
    #
    # result = poisson_editing(source_array, target_array, mask_array, offset=(-365, 0), mixing=True)
    # im_result = Image.fromarray((result * 255).astype(np.uint8))
    # im_result.save('rainbow/risultati_finali/result_mixed.jpg', 'JPEG')
    # im_result.show()

    '''Local color changes'''
    source = Image.open('flower/source.jpg')
    target = Image.open('flower/source.jpg').convert('L')
    mask = Image.open('flower/mask.jpg')
    mask = ImageOps.grayscale(mask)

    source = source.resize((source.size[0] // 4, source.size[1] // 4))
    target = target.resize((target.size[0] // 4, target.size[1] // 4))
    mask = mask.resize((mask.size[0] // 4, mask.size[1] // 4))
    save_images("flower", source, target, mask)

    source_array = np.array(source)
    target_array = np.array(target)
    mask_array = np.array(mask)

    result = poisson_editing(source_array, target_array, mask_array, mixing=False)
    im_result = Image.fromarray((result * 255).astype(np.uint8))
    im_result.save('flower/risultati_finali/result.jpg', 'JPEG')
    im_result.show()
