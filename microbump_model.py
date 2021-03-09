import numpy as np
import matplotlib.pyplot as plt
import cv2


def stack(filepath):  # return 3D microbump raw image
    s = 0
    total = 0
    for i in range(1000):
        n = str(i+1)
        img = cv2.imread(filepath + n + '.tiff', 0)  # 原始檔讀取
        if img is None:  # filename unreachable
            continue
        total += 1
    count = 0
    for i in range(1000):
        n = str(i+1)
        img = cv2.imread(filepath + n + '.tiff', 0)  # 原始檔讀取
        if img is None:  # filename unreachable
            continue
        if count == 0:
            s = np.empty((img.shape[0], img.shape[1], total))
            s[:, :, count] = np.array(img)
        else:
            s[:, :, count] = np.array(img)
        count += 1
    return s


def main():
    area = stack('F:/J/TSMC/smooth/N17_area1_initial/')  # read file
    print(area.shape)
    bump = np.empty((100, 1, 60, 60, 60))  # split into bump by bump
    count = 0  # the number of microbump
    start = 0
    end = 0
    sample = 0
    structure_check = []
    for depth in range(area.shape[2]):
        pres = np.sum(area[:, 6:-6, depth])
        if pres > 0 and start == 0:
            start = depth
        elif start != 0 and pres == 0:
            end = depth
        if start != 0 and end != 0:
            if (end - start) < 40:
                start, end = 0, 0
                structure_check.append(0)
                continue
            structure_check.append(1)
            line = area[:, 6:-6, start:end]
            print(line.shape)
            head = 0
            tail = 0
            for length in range(line.shape[0]):
                obj = np.sum(line[length, :, :])
                if obj > 0 and head == 0:
                    head = length
                elif head != 0 and obj == 0:
                    tail = length
                if head != 0 and tail != 0:
                    if (tail - head) < 30:
                        head, tail = 0, 0
                        continue
                    m1 = (tail+head)//2
                    m2 = (end+start)//2
                    unit = area[m1-30:m1+30, :, m2-30:m2+30]
                    print(unit.shape)
                    origin = (60-unit.shape[1])//2
                    bump[count, :, :, origin:origin+unit.shape[1], :] = unit
                    count += 1
                    sample += 1
                    head, tail = 0, 0
                    continue
            start, end = 0, 0
            continue
    target = np.load('N17_initial.npy')
    plt.matshow(target)
    plt.show()
    print(sample)
    print(structure_check)


if __name__ == '__main__':
    main()