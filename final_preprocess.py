import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import scipy.misc
from skimage import morphology, color, data, filters
from sklearn import linear_model
from PIL import Image
from skimage.feature import peak_local_max
from skimage.exposure import match_histograms


def stack():  # return 3D microbump raw image
    s = 0
    mean = []
    total = 0
    for i in range(1000):
        n = str(i+1).zfill(4)
        img = cv2.imread('F:/J/TSMC/N12_reflow/N12-Area2-TIFF/'+n+'.tiff', 0)  # 原始檔讀取
        if img is None:  # filename unreachable
            continue
        total += 1
    count = 0
    for i in range(1000):
        n = str(i+1).zfill(4)
        img = cv2.imread('F:/J/TSMC/N12_reflow/N12-Area2-TIFF/'+n+'.tiff', 0)  # 原始檔讀取
        if img is None:  # filename unreachable
            continue
        if count == 0:
            s = np.empty((img.shape[0], img.shape[1], total))
            s[:, :, count] = np.array(img)
            mean.append(np.mean(img))
        else:
            s[:, :, count] = np.array(img)
            mean.append(np.mean(img))
        count += 1
    return s, np.argmax(mean[:100])


def find_slice(image_3d, plane):
    mean = []
    if plane == 'yz':
        for i in range(image_3d.shape[1]):
            if i == 0:
                mean.append(np.max(image_3d[:, i, :]))
            else:
                mean.append(np.max(image_3d[:, i, :]))
    elif plane == 'xz':
        for i in range(image_3d.shape[0]):
            if i == 0:
                mean.append(np.max(image_3d[i, :, :]))
            else:
                mean.append(np.max(image_3d[i, :, :]))
    return np.argmax(mean)


def load_reference():
    count = 0
    ref = 0
    while True:
        img = cv2.imread('F:/J/ref/'+str(count+1)+'.tiff', 0)  # 原始檔讀取
        if img is None:  # filename unreachable
            break
        count += 1
    for i in range(count):
        img = cv2.imread('F:/J/ref/' + str(i+1) + '.tiff', 0)
        if i == 0:
            ref = np.empty((img.shape[0], img.shape[1], count))
            ref[:, :, i] = np.array(img)
        else:
            ref[:, :, i] = np.array(img)
    return ref


def blur_solve(cut):
    grad_x, grad_y, grad_z = np.gradient(cut, 0.2)
    grad_x, grad_y, grad_z = np.absolute(grad_x), np.absolute(grad_y), np.absolute(grad_z)
    grad = np.maximum(grad_x, grad_y)
    grad = np.maximum(grad, grad_z)
    grad[grad < 50] = 0
    grad[cut == 0] = 0
    coor = np.argwhere(grad > 0)
    changed = np.zeros(cut.shape)
    for x, y, z in coor:
        kernel = cut[x - 2:x + 3, y - 2:y + 3, z - 2:z + 3]
        if 0 in kernel:
            changed[x, y, z] = np.max(kernel)
    coor = np.argwhere(changed > 0)
    for x, y, z in coor:
        cut[x, y, z] = changed[x, y, z]
    return cut


def main():
    struct_3d, max_bump = stack()
    print('loading finished')

    # xy plane rotate
    ref = struct_3d[:, :, max_bump]
    coor = np.argsort(ref, axis=None)[-1001:-1]  # location of pixels
    coor_2d = np.zeros((1000, 2))
    for i in range(1000):
        coor_2d[i][0] = coor[i]//ref.shape[1]
        coor_2d[i][1] = coor[i] % ref.shape[1]
    reg = linear_model.LinearRegression()
    reg.fit(coor_2d[:, 0].reshape(-1, 1), coor_2d[:, 1].reshape(-1, 1))
    a = math.degrees(math.atan(reg.coef_))  # angle calculation
    for i in range(struct_3d.shape[2]):  # rotate
        img = Image.fromarray(struct_3d[:, :, i])
        img = img.rotate(-a)
        struct_3d[:, :, i] = np.array(img)
    print('xy calibration finished')

    # yz plane rotate
    mid = find_slice(struct_3d, 'yz')
    ref = struct_3d[:, mid, 45:134]
    coor = np.argsort(ref, axis=None)[-1001:-1]  # location of pixels
    coor_2d = np.zeros((1000, 2))
    for i in range(1000):
        coor_2d[i][0] = coor[i] // ref.shape[1]
        coor_2d[i][1] = coor[i] % ref.shape[1]
    reg = linear_model.LinearRegression()
    reg.fit(coor_2d[:, 0].reshape(-1, 1), coor_2d[:, 1].reshape(-1, 1))
    a = math.degrees(math.atan(reg.coef_))  # angle calculation
    for i in range(struct_3d.shape[1]):  # rotate
        img = Image.fromarray(struct_3d[:, i, :])
        img = img.rotate(-2.3)
        struct_3d[:, i, :] = np.array(img)
    print('yz calibration finished')

    # xz plane rotate
    ref = struct_3d[find_slice(struct_3d, 'xz'), :, :]
    coor = np.argsort(ref, axis=None)[-1001:-1]  # location of pixels
    coor_2d = np.zeros((1000, 2))
    for i in range(1000):
        coor_2d[i][0] = coor[i] // ref.shape[1]
        coor_2d[i][1] = coor[i] % ref.shape[1]
    reg = linear_model.LinearRegression()
    reg.fit(coor_2d[:, 1].reshape(-1, 1), coor_2d[:, 0].reshape(-1, 1))
    a = math.degrees(math.atan(reg.coef_))  # angle calculation
    print(a)
    for i in range(struct_3d.shape[0]):  # rotate
        img = Image.fromarray(struct_3d[i, :, :])
        img = img.rotate(a)
        struct_3d[i, :, :] = np.array(img)
    print('xz calibration finished')
    # for i in range(struct_3d.shape[2]):
    #     plt.imsave('F:/J/N12-Part4-crop/'+str(i+1)+'.tiff', vmin=0, vmax=255, cmap='gray', arr=struct_3d[:, :, i])
    #
    # img = cv2.imread('F:/J/N12-Part4-crop/'+str(max_bump+1)+'.tiff', 0)
    img = struct_3d[:, :, max_bump]
    img = filters.rank.median(img.astype(np.uint8), morphology.disk(3))
    h, w = img.shape[0], img.shape[1]//5
    x, y = w*2, 0
    img = img[y:y+h, x:x+w]  # 中央1/5區塊
    bump = np.zeros((img.shape[0]))
    for i in range(img.shape[0]):
        bump[i] = np.sum(img[i, :])
    bump = np.argsort(bump)[-53]
    gradient = filters.rank.gradient(img, morphology.disk(3))
    gradient_c = gradient[bump, :]
    coor = peak_local_max(gradient_c, min_distance=3).reshape(1, -1)
    sort = np.flipud(np.argsort(gradient_c[coor])[0, :])[:4]
    sort = coor[0][sort]
    left_b = np.min(sort)
    right_b = np.max(sort)
    w = right_b - left_b
    x = x + left_b

    reference = load_reference()  # reference of histogram matching
    target = struct_3d[y:y + h, x-3:x + w+3, :]
    matched = match_histograms(target, reference, multichannel=False)
    print('histogram matching finished')
    result = filters.rank.median(matched[:, :, max_bump].astype(np.uint8), morphology.disk(3))
    result[result < 110] = 0
    left = 0
    right = 0
    for i in range(matched.shape[1]):  # find the left margin of microbump
        if np.sum(result[:, i]) > 0:
            left = i
            break
    for i in range(matched.shape[1]):  # find the right margin of microbump
        if np.sum(result[:, matched.shape[1] - i - 1]) > 0:
            right = matched.shape[1] - i - 1
            break
    matched = matched[:, left-3:right+3, :]
    for i in range(matched.shape[2]):
        result1 = filters.rank.median(matched[:, 5:-5, i].astype(np.uint8), morphology.disk(3))
        result2 = filters.rank.median(matched[:, :5, i].astype(np.uint8), morphology.disk(3))
        result3 = filters.rank.median(matched[:, -5:, i].astype(np.uint8), morphology.disk(3))
        result1[result1 < 110] = 0
        result2[result2 < 70] = 0
        result3[result3 < 70] = 0
        matched[:, 5:-5, i][result1 == 0] = 0
        matched[:, :5, i][result2 == 0] = 0
        matched[:, -5:, i][result3 == 0] = 0
    matched[:, 10:-10, :] = blur_solve(matched[:, 10:-10, :])
    m1 = matched[:, 5:-5, :]
    matched[:, 5:-5, :][m1 < 110] = 0
    matched[matched > 170] = 200
    matched[(matched >= 158) & (matched <= 170)] = 160
    matched[(matched < 158) & (matched > 0)] = 120
    # matched[(matched < 110) & (matched > 0)] = 80
    # matched[(matched < 110) & (matched > 0)] = 80
    print('saving image')
    for i in range(0, matched.shape[2]):
        plt.imsave('F:/J/TSMC/N12_reflow/N12-Area2_reflow_crop/' + str(i+1) + '.tiff', vmin=0, vmax=255, cmap='gray', arr=matched[:, :, i])


if __name__ == '__main__':
    main()