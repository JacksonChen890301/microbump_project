# Barlow Twins  

## 使用目的
<font size=3>在microbump的資料中，initial和reflow都有照CT的樣本很少。如果想要透過initial狀態去推估reflow後的電阻或電阻提升量，資料量顯然不足。為了彌補此問題，因此嘗試用self-supervised的方法將資料的使用效益最大化，測試是否能克服資料上的問題。</font>  

## Barlow Twins概述  
<font size=3>自監督學習中有一種常用的方法，是透過對同一張圖片加上一些變換(例如旋轉、亮度調整等等)，然後要求模型能夠辨識出兩張經過不同變換的圖片是來自同源。這個過程能幫助模型在不使用label的情況下，依舊能從圖像中學習到一些特徵。並且再透過這個pre-trained好的模型，運用在其他的任務上。<br>
    而以上的方法正是Barlow Twins(BT)的核心。BT模型由CNN作為Encoder將圖片降維，MLP作為Projector負責處理相似度的問題。最後，兩張經過不同Augmentation的圖片會被此模型分別輸出為向量。這兩個向量feature之間的covariance matrix會用於loss function，對角線上的covariance要傾向越接近1越好，而其餘的covariance則是越接近0越好。這樣一來，既能保證讓整個模型趨向使同一組的vector最相似，而且每個vector上的component又能盡可能線性獨立。</font>  

<div align=center><img src="https://i.imgur.com/RhrTTCY.png" width="100%" alt="img01"/></div>  

## 模型結構  
<div align=center><img src="https://i.imgur.com/1Dy0epv.png" width="50%" alt="img01"/></div>  
<font size=3></font>  
