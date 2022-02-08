---
title: 'Microbump前處理紀錄'
tags: 'LAB'
---
# Microbump前處理紀錄  

## 問題概述  
<font size=3>資料來源為非破壞性檢測(CT)所得到的影像，目的是要把2D切片影像重建回3D，並去除背景，僅留下*microbump*本體。最後，根據*grayscale*分布將不同的材料分類，以供後續*ANSYS*或是AI工具處理。</font>  
<center><img src="https://user-images.githubusercontent.com/55709819/132184994-79661609-f2e0-4c89-83ae-7d05acc236ad.png" width="70%" alt="img01"/></center>  
<center><font  size=2>microbump的CT原圖</font></center>  

## 旋轉校正  
<font size=3>由於取像的時候，樣本會有歪斜的狀況。因此，為了方便後續處理，必須校正此誤差。從下圖可以看出，在此截面下，該列*microbump*有明顯的偏移現象。</font>  
<center><img src="https://i.imgur.com/Xj9jOLK.png" width="30%" alt="img01"/></center>  

<center><font size=2>右圖用一threshold保留microbump本體</font></center><br>  

<font size=3>校正方法: 先對xy截面取出一影像(要挑*microbump*截面積最大的截面)，然後挑出*grayscale*最大的1000個pixel，並記錄他們的x-y座標。最後，對這1000個點做線性回歸，並求出此回歸線之斜率，此斜率正是偏移角度的tan值。之後分別對xz, yz截面也做一樣的步驟，即可完成校正。</font>  
```
ref = bump_3d[:, :, max_slice]  # 挑出截面積最大的slice
coor = np.argsort(ref, axis=None)[-1001:-1]  # 挑出灰度最高的1000個點
coor_2d = np.zeros((1000, 2))

# 因為回歸線水平時斜率為0，我們把microbump排列方向定義為x軸
for i in range(1000):
    coor_2d[i, 0] = coor[i]//ref.shape[1]  # x座標
    coor_2d[i, 1] = coor[i] % ref.shape[1]  # y座標
    
from sklearn import linear_model
reg = linear_model.LinearRegression()  # 建立線性回歸模型
reg.fit(coor_2d[:, 0].reshape(-1, 1), coor_2d[:, 1].reshape(-1, 1))
a = math.degrees(math.atan(reg.coef_))  # 透過斜率計算旋轉角度

from PIL import Image
for i in range(bump_3d.shape[2]):  # 套用旋轉角度
    img = Image.fromarray(bump_3d[:, :, i])
    img = img.rotate(-a)
    bump_3d[:, :, i] = np.array(img)
```
<center><img src="https://i.imgur.com/JH6uzYj.png" width="70%" alt="img01"/></center>  

## 中值濾波器: Median Filter  
```
cv2.medianBlur(img, ksize=3)
```
<font size=3>用於降噪並保持邊緣特性，效果如下</font>  
![median_filter](https://i.imgur.com/Dhf5zkS.png)
<font size=3>其計算過程，根據維基百科的例子:
x是待處理的數組，設定filter size為3，遇到邊緣則重複該數字
```
x = [2 80 6 3]
y[1] = Median[2 2 80] = 2
y[2] = Median[2 80 6] = Median[2 6 80] = 6
y[3] = Median[80 6 3] = Median[3 6 80] = 6
y[4] = Median[6 3 3] = Median[3 3 6] = 3
```
於是經過中值濾波後
`y = [2 6 6 3]`
</font>  

## 裁切: cropping  
<font size=3>經過旋轉校正後，可以將多餘的背景部分捨棄。這邊使用梯度的方法去抓microbump的左邊界和右邊界，差不多會是兩邊銅導線的位置。</font>  
```
from skimage import filter, morphology
# 挑一個截面積最大的截面最為img
gradient = filters.rank.gradient(img, morphology.disk(3)) # 計算梯度，範圍是半徑為3的圓
# 一樣挑截面積夠大的row當成center
gradient_c = gradient[center, :]

from skimage.feature import peak_local_max
# 尋找梯度的局部最大值
coor = peak_local_max(gradient_c, min_distance=3).reshape(1, -1)
# 找出前幾大的局部最大值
sort = np.flipud(np.argsort(gradient_c[coor])[0, :])[:4]
sort = coor[0][sort]
# 根据column的index大小決定左右界
left_b = np.min(sort)
right_b = np.max(sort)
```
<center><img src="https://i.imgur.com/E2A5BQY.png" width="70%" alt="img01"/></center>  

## histogram matching  
<font size=3>由於儀器本身或人為操作問題，不同批的ct影像可能在灰度值的分布上會有落差。為了將所有資料的亮度標準化，這裡需要使用*histogram matching*來校正。其原理是透過機率累積函數，計算pixel隨著grayscale逐漸數量增加的過程。我們已知每批資料中microbump的體積和各項材質的占比幾乎一致。因此，無論每批資料的亮度如何，在機率累積的視角來看，同樣的材質在不同樣本中，勢必會佔據相同的比例。所以我們只需要先將一批資料作為參考，將另一批資料中各個分布區間的pixel值平移到跟參考值相等就好。</font>  

![](https://i.imgur.com/E7SHiOC.png)
<center><font  size=2>histogram matching圖示</font></center>  

```
from skimage.exposure import match_histograms
matched = match_histograms(target, reference, multichannel=False)
```
![](https://i.imgur.com/WbIN04R.png)
<center><font  size=2>處理後的分布情況</font></center><br>  

<font size=3>如果只是要給AI或機器學習工具使用，到這裡就差不多做完前處理了，只差把每一個microbump分別裁出來然後固定尺寸即可(大約每顆會是一個64x64x64的立方體)後面的步驟則是ANSYS計算才需要做的前處理。</font>  



