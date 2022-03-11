# 第一章 簡介

## 1.1 前言:
<font size=3>隨著半導體產業蓬勃發展以及更加輕薄多功能的電子產品問世，電子元件的密度與尺寸被不斷壓縮。根據摩爾定律所述，積體電路上可容納的電晶體數目，約每隔兩年便會增加一倍。電晶體尺寸已從微米等級不斷推進至目前的奈米等級。在此過程中，傳統二維大型積體電路逐漸到達極限，而被三維積體電路 (3D Integrated Circuit, 3D IC)堆疊技術所取代。在此技術中，用於垂直連接晶片的銲錫微凸塊(solder microbump)，不僅可以增加 I/O 數目，還有著低成本的優勢。如圖1-1所示[1]，銲錫微凸塊的製造，需要透過迴銲(reflow)在高溫爐中以高於銲錫熔點的溫度加熱大約一分鐘，再經由熱壓(thermo-compression)步驟，以接合器在迴銲溫度下加壓數秒以達成接合晶片的效果。<br>
  
銲錫微凸塊相較於傳統覆晶銲錫凸塊，尺寸僅有20微米，銲錫所占體積百分比大幅降低，能直接降低產品的尺寸。然而，焊料的減少也同時會引起許多可靠性疑慮，包含了焊接點產生介面金屬共化物(Intermetallic Compound, IMC)[2]以及凸塊下金屬材(Under Bump Metallization, UBM)周圍經由銲錫擴散及潤濕，所導致的頸縮(necking)或空洞(voiding)現象[3]，如圖1-2。為了減緩、甚至進一步避免以上問題對元件的損害，銲錫微凸塊的結構及材料仍有探討及改善的空間。</font>  
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156309935-cc637c74-b95a-4f84-b44e-7133635ee0bf.png width="500"></div>
<div align=center><font size=3>圖1-1、迴銲流程示意圖[1]</font></div>  
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156310079-91be08e4-49aa-4140-a299-c0fd9ee85468.png width="500"></div>
<div align=center><font size=3>圖 1-2、銲錫微凸塊截面影像(a)剛連接時(b)迴銲30分鐘(c)迴銲2小時</font></div>

## 1.2 研究計畫摘要
<font size=3>銲錫微凸塊(solder microbump)通常是以無鉛銲錫(Lead-free solder)為材料，在3D IC封裝中，做為晶片垂直堆疊的橋樑。此技術可大幅縮小IC的體積，並具有密度大、低感應、低成本、散熱能力佳等優點。然而此結構中的銲錫會因受熱而變形或產生介金屬化合物(Intermetallic Compound, IMC)，導致元件的機械性質及導電能力受損，影響晶片的性能及壽命。<br>
  
由於研究樣本在電路設計限制下，只能做到全局性的量測。因此，必須對實體晶片中的銲錫微凸塊建模，並使用有限元素法計算局部電阻，才得以進一步探討銲錫變形對元件的影響。為了改善有限元素法冗長的計算量，本計畫採用深度學習中的卷積神經網路(Convolutional Neural Network, CNN)[4]以及Transformer模型[5]做為替代，並透過常用於分析深度學習模型的分層相關性傳播(Layer-wise relevance propagation, LRP)[6]，確保模型的可靠性。最終，訓練出高準確率與計算速度，並有合理判斷依據的模型，以拓展工業上對產品即時分析的手段及可能性。</font>  

# 第二章 文獻回顧  

## 2.1 Sn-Ag銲錫發展
<font size=3>在覆晶封裝製程中，銲錫與UBM必須有良好的潤濕性，才能確保接合的品質。而傳統銲料常使用的錫鉛(Sn-Pb)合金，由於有Pb的添加，使其與Cu材質UBM的潤濕角(wetting angle)僅有11˚。此現象使得反應產生的Cu6Sn5介金屬化合物與Sn-Pb銲錫的界面能降低[7]，減少了銲錫與Cu6Sn5發生剝離現象(spalling)[8]的可能性。但考量到鉛會對人體及環境造成危害，歐盟於西元2006年頒布「限用有害物質指令」(RoHS)[9]，使得電子封裝產業必須開發無鉛銲錫材料做為替代。其中，Sn-Ag二元合金銲錫具備良好拉伸、潛變、疲勞等機械性質，因此被高度重視。<br>
  
Sn-Ag銲錫並非完美的替代品。因為其熔點較Sn-Pb高，使得Sn-Ag在迴銲過程中更容易與Cu反應，產生Cu6Sn5介金屬化合物。而且Sn-Ag銲錫與Cu的潤濕性也較差，易導致Cu6Sn5剝離。為克服以上問題，可在Cu和銲錫之間加上Ni金屬層，以增加銲錫的潤濕性。且由於Ni與銲錫間的反應慢[10]，因此可作為屏障，減少銲錫與Cu的反應。然而，即便透過Ni層改善Sn-Ag銲錫的性質，銲料仍會為了形成介金屬化合物以降低系統自由能，在迴銲時擴散到UBM周圍[11]，如圖2-1。由於銲料體積小，擴散時易形成頸縮及空孔，造成微凸塊的電性與機械性質衰弱。</font>
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156223977-5ae80905-2a5f-4a3e-a8be-6250b549e240.png width="500"></div>
<div align=center><font size=3>圖2-1、(a)剛生產出之銲錫微凸塊(b)微凸塊在260°C下迴銲20分鐘[12] 圖(b)可見銲錫中央已經完全空乏</font></div>  

## 2.2 深度學習於三維斷層掃描影像之應用  
<font size=3>深度學習(Deep Learning)為機器學習(Machine Learning)的其中一個分支，該模型以人工神經網路(Artificial Neural Network)為基礎，能夠從大量資料中學習重要特徵並輸出給定任務的答案。其中，最常見的應用便是在電腦視覺(Computer Vision)領域，為圖像辨識、分割等問題提供了更優秀的解方。而卷積神經網路(Convolutional Neural Network, 以下簡稱CNN)便是該領域中主流的模型架構。CNN的雛型為1998年由Yann LeCun等人提出的網路架構LeNet[4]，但由於當時運算能力的限制，神經網路的研究曾沉寂多時。之後隨著硬體技術的進步，從2012年ImageNet圖像分類競賽冠軍AlexNet[13]開始，CNN模型的相關論文便不斷湧現，模型性能也在持續增長。<br>
  
近年來，CNN技術已臻成熟，在自駕車、人臉辨識或是工業上的產品檢測都有一席之地，而醫學中常用的電腦斷層攝影也受益於此技術的發展。醫學影像由於其結構複雜且對比度低，在臨床上難以使用傳統的影像分析技術做為診斷的輔助。因此，深度學習成為此類問題的主要研究方向，並有效的適應於多種斷層攝影使用場景上。例如，Huang等人[14]透過3D卷積神經網路從電腦斷層影像中檢測患者是否有肺結節病徵。而Gao等人[15]也使用相同基礎的模型，在阿茲海默症的判別上達到了85.2%的準確率。</font>  

# 第三章 研究方法  

## 3.1 斷層掃描 (Computed Tomography, CT)取像及前處理
<font size=3>本計畫研究之晶片由於電路設計的限制，在實驗上只能對特定區域的連續40顆或400顆串聯的微凸塊做電性量測，如圖3-1所示。其中，Daisy 40以及Daisy 400即分別代表上述之40顆和400顆微凸塊對應到晶片上的實際位置。因此，量測值只能得到整體性的資訊，而無法顯現各個微凸塊的狀況。為此，實驗端透過非破壞性的斷層掃描，取得微凸塊的立體完整結構供後續分析，如圖3-2所示。</font>  
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156310369-99a067d7-8d5f-464c-95dc-00bfb50c6286.png width="500"></div>
<div align=center><font size=3>圖3-1: 晶片結構以及可量測區域示意圖，綠色虛線代表電流路徑。</font></div>
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156310480-1ac258a3-e10b-486a-947e-139c67b165f1.png width="500"></div>
<div align=center><font size=3>圖3-2、(a)斷層掃描方向(b)掃描之截面原圖</font></div><br>

<font size=3>斷層掃描圖像在完成三維的堆疊後，即可得到微凸塊電路之立體結構。其最小的單位即為在空間中均勻分布的立方體素(voxel)，所以也可被視為三維矩陣。然而，斷層掃描的取像過程中常有人為或儀器的誤差，造成各晶片樣本的朝向以及影像亮度不同。為避免樣本間差異造成模型之偏誤，必須盡可能使所有樣本處於同一標準下。因此，歪斜的晶片須對齊由三維矩陣之長寬高所構成的笛卡兒座標系，並透過直方圖匹配(histogram matching)[16]，將所有影像的灰度(grayscale)變換至同一區間內。整個前處理流程大致如圖3-3。</font>  
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156310800-7604929a-9047-4b71-b0e7-840673212770.png width="500"></div>
<div align=center><font size=3>圖3-3、斷層掃描圖像前處理流程</font></div>

## 3.2 有限元素法電阻模擬  
<font size=3>經過3.1的前處理流程，可以得到去除所有背景，僅留下微凸塊電路本身的三維矩陣。為了增加資料量以及降低計算複雜度，電路上的所有微凸塊都會被分別裁剪。最後，再透過串聯電路的特性，將各個微凸塊計算出的電阻加總，即可得知整個電路的模擬電阻值，並與實際量測值做對照。<br>
                                                                                                                                                 >
每顆微凸塊所形成的三維矩陣會先以一階四面體元素建模，後續再使用更多節點的元素修正。由於微凸塊在斷層影像中大致上只能看出銅、鎳以及錫銀合金三種材料。因此，在計算前需根據灰度調整閥值，將微凸塊矩陣上的體素先行分類，以便後續套用各材料對應的電阻率，如表3-1。<br>
  
本計畫有限元素法求解過程所考慮的邊界條件僅有電流及電壓。其中，電流設定為1.5安培，並施加於導線入口，導線出口的邊界條件則設置為電壓等於零，如圖3-4。以上的計算皆在有限元素法模擬軟體ANSYS上運行，並使用其提供的元素類型，SOLID226。此元素適用於三維的電磁場、熱場、電場、壓電、結構場等耦合分析，為二階的三維六面體。其中，每個單元有20個節點，每個節點有6個自由度(UX、UY、UZ、TEMP、VOLT、MAG)，如圖3-5。</font>
<div align=center><font size=3>表3-1、模擬所使用之電阻率</font></div>
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156311044-05312ccd-45bf-4247-bf26-e174c6d02537.png width="500"></div>
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156311268-c397e73f-e542-421a-b64b-5d882d917732.png width="500"></div>
<div align=center><font size=3>圖3-4、邊界條件示意圖</font></div>
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156311559-c526390d-3a05-44cd-b588-a7e34a1112fd.png width="500"></div>
<div align=center><font size=3>圖3-5、SOLID226幾何結構</font></div>

## 3.3 深度學習模型設計  
<font size=3>本計畫以三維卷積神經網路（3D Convolutional Neural Network以下簡稱為CNN）為基礎，打造更加快速、簡便的模型，重現ANSYS模擬之結果。CNN憑藉其在圖像處理任務上優異的準確度，已被廣泛的應用在斷層造影相關問題。例如，肺結節病徵判別[14]以及阿茲海默症診斷[15]。<br>
  
雖然CNN已是各領域研究中的常客，它仍有一些已知的缺點。(1)它無視圖樣中前景與背景之關係，所有像素皆平等的進入運算，造成計算量的浪費；(2)高階特徵比起低階特徵的泛用性低非常多，造成計算量以及儲存空間的浪費；(3)傳統CNN對於空間上距離較遠之特徵的相關性難以評估。為了彌補以上的不足，本計畫使用由Facebook及UC Berkeley聯合發表的Visual Transformer(以下簡稱VT)架構作為補強。該論文的核心思想強調了圖像和語言的相似性，認為影像可拆解為前景、背景等多個物件，正如同語句由多個單字所構成。因此，VT融合了近年在圖像描述、聊天機器人、語音辨識以及機器翻譯等各領域大放異彩的Transformer模型以突破目前CNN模型的瓶頸。</font>  
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156311771-ee6d9666-5da9-4654-b814-367722d5325b.png width="500"></div>
<div align=center><font size=3>圖 3-6、本計畫使用之神經網路結構圖</font></div>

## 3.4 模型訓練細節  
<font size=3>供模型訓練之資料共包含了四個晶片，每個晶片上可量測的電路有400顆微凸塊，並且每個試片都會在260˚C下迴銲至少30分鐘後，再測量一次電阻。因此，理論上會有3200顆微凸塊的資料。然而數據並不是完好無損，所以經由前處理完成資料清洗後，能用於模擬的微凸塊僅存1699顆。其中，剛銲接完成之初始狀態微凸塊有1040顆，迴銲30分鐘以上之微凸塊則有659顆。<br>
  
為了使深度學習模型有足夠泛化能力並充分收斂，樣本會被切割為訓練及驗證組，比例約為四比一。訓練組的資料會幫助模型學習、將誤差收斂至最小，而驗證組則用於檢查前者所訓練出的模型是否有足夠的準確度，並防止訓練過程發生過擬合的現象。經過反覆的訓練及參數調整，得到驗證組預測誤差最低的模型後，再以額外保留的90筆測試組資料做為最後的評估手段，藉此保障模型的泛用性，詳細資料分布如表3-2。<br>
  
訓練模型所使用的損失函數為均方誤差(Mean square error, MSE)，使用Adam(adaptive moment estimation)優化器，並設定正則化參數為0.003。訓練用的硬體設備為Tesla V100 GPU，以64為batch size訓練150個epoch。</font>
<div align=center><font size=3>表3-2、資料分布</font></div>
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156312033-cd38b46a-780f-49b4-9498-a01437c2213f.png width="500"></div>
<div align=center><font size=3>* Testing所使用的90顆微凸塊為電路上的一完整區域</font></div>

# 第四章 結果與討論

## 4.1有限元素法模擬結果
<font size=3>每個微凸塊在ANSYS上建模並完成計算，大約需要一個小時。圖4-1為電壓分布之計算結果。完成樣本共四個晶片的微凸塊計算後，透過將每個區域400顆微凸塊的模擬電阻加總，即可推知完整電路的模擬結果。考慮到資料有缺損或雜訊嚴重之情形，電路中無法求出模擬數值的微凸塊則會以該電路剩餘微凸塊的平均值作為代替，如圖4-2。其中，圖(a)為剛完成銲接之初始狀態，圖(b)為迴銲30分鐘後之狀態，空白區域為缺失的微凸塊資料。<br>
  
模擬完成並與實際電路量測資料比對後，可發現模擬計算之電阻值大約為實際電阻值的60%，呈現固定比例差距的關係，如表4-1。且在初始及長時間迴銲後兩種電阻差異高達20%的樣本中，模擬上皆取得相似的誤差，由此可確定有限元素法之計算結果能有效反映樣本間的差距。</font>
<div align=center><font size=3>表4-1、其中兩片初始及迴銲狀態資料較完整之晶片模擬結果</font></div>
<div align=center><img src=https://user-images.githubusercontent.com/55709819/156312350-6e8c4ee6-f25e-4963-91b5-6f13f945e723.png width="500"></div>
<div align=center><font size=3>* Ratio = Simulation / Experiment</font></div>

<div align=center><img src=https://user-images.githubusercontent.com/55709819/156312552-8323a3e9-f2a2-4aaf-9736-9a2aac5f32fa.png width="500"></div>
<div align=center><font size=3>圖4-1、ANSYS計算之微凸塊電壓分布</font></div>

<div align=center><img src=https://user-images.githubusercontent.com/55709819/156312680-352455f5-b6e6-4eb8-aaab-df64e12a11d0.png width="500"></div>
<div align=center><font size=3>圖4-2、ANSYS計算之電阻分布</font></div>

## 4.2深度學習電阻預測結果
<font size=3>本模型的預測能力在驗證組及測試組上皆有優良表現，如圖4-3所示。其中，橫軸代表ANSYS模擬值，縱軸代表模型預測值。每個點都是一顆微凸塊的計算結果，並分別以紅、藍色代表長時間迴銲後以及初始狀態下之微凸塊。若以均方根誤差(Root Mean Square Error, RMSE)作為評價誤差的標準，兩組資料上的預測誤差僅有約0.14mΩ，相當於平均模擬電阻值的1%。此結果反映該模型已訓練出有效的判斷機制，得以擬合有限元素法的計算結果。</font>

<div align=center><img src=https://user-images.githubusercontent.com/55709819/157802412-04252e73-c364-4bcb-aeb8-05cbb75569f2.png width="500"></div>

<font size=3>若將本模型與卷積層結構相同的CNN模型做比較，可發現兩者雖然在驗證組上表現相近，但是在測試組上的表現有著明顯落差。其中，普通的CNN模型在測試組上的RMSE約為0.21mΩ。因此，可推知使用了VT架構的模型在泛用性上略勝一般的CNN模型一籌。</font>

<div align=center><img src=https://user-images.githubusercontent.com/55709819/156312851-0a1f91fd-ef5c-4f05-a06d-be15932ae0bf.png width="700"></div>

## 4.3分層相關性傳播(Layer-wise relevance propagation)分析
<font size=3>類神經網路憑藉其高效能，近年來在各領域已被廣泛運用。然而，因為模型本身的高複雜度，造成解釋力低下，甚至被比喻為黑箱。為解決這類模型不透明的現象，目前已發展出多種手段能初步分析其下判斷的理由，並藉此審視模型是否有不合邏輯或帶有偏見等狀況。為此，本計畫使用分層相關性傳播方法(Layer-wise relevance propagation, 以下簡稱LRP)，彌補此種模型預測依據存疑的缺點。<br>
  
LRP最早發表於Bach et al.(2015)[6]，一般用於解釋影像辨識模型。LRP可以計算出輸入的影像資料中，每一個像素(pixel)對於辨識結果的重要性(relevance)，最終由熱度圖(heatmap)呈現重要性隨像素的分布。LRP演算法相當直觀，透過反向傳播，將輸出層的結果根據各神經元的權重分配至前一層，並一路回推到輸入層，如圖 4-2[5]所示。因此，本研究可利用LRP計算出的相關性分布，得知銲錫微凸塊三維結構中各體素(voxel)對電阻的貢獻。<br>
  
圖4-3為根據LRP分析本模型的結果所繪製之熱度圖。其中，圖(a)(b)(c)(d)分別代表不同截面，紅色熱度點的分布則表示該體素和電阻的相關性。經由此圖，可得出以下結論。(1)背景資訊對預測結果幾乎毫無貢獻，符合正常邏輯。(2)相較於銅，中央銲錫的重要性明顯較高，且集中在其邊緣。綜上，本模型確實從訓練資料中歸納出一套機制以過濾多餘的資訊。</font>

<div align=center><img src=https://user-images.githubusercontent.com/55709819/156313188-36df186e-05e1-496c-9e18-6c7ae8c72669.png width="700"></div>
<div align=center><font size=3>圖4-4、銲錫微凸塊中體素與電阻之相關性</font></div>

# 第五章 結論
<font size=3>本計畫透過深度學習模型，簡化了常用於分析物理現象的有限元素法在銲錫微凸塊上的計算過程。將原本在ANSYS上需要約一小時的模擬時間縮小至僅需數秒。此成果主要歸功於深度學習模型的輸入不需要複雜的三維元素建模，也不需要對資料做太多的人為處理，就能夠透過深度學習可有效捕捉潛在特徵的特性，取代原本須繁複計算的模擬流程。並且，在分層相關性傳播等工具的使用下，能確保模型的可解釋性，以便對模型的計算結果歸因。若進一步將本計畫之方法或模型優化，可部屬於工業上，對產品做即時量測及分析，增加品質管理的手段。<br>
  
本計畫仍有不少繼續發展的空間。例如，在數據充足的情況下，可透過初始狀態下的微凸塊形貌，預測經過一固定時間迴銲後之電阻上升量，或是預測一個微凸塊在高溫下使用多久後會完全斷裂。以上的研究主題除了提供更多對產品的分析方向之外，也有助於微凸塊設計上的改善。</font> 


# 參考文獻
[1] 	Jang, J. W., et al. "High-lead flip chip bump cracking on the thin organic substrate in a module package." Microelectronics Reliability 52.2 (2012): 455-460.<br>
[2] 	Y.W. Chang, C. Chen, T.C. Chang, C.J. Zhan, J.Y. Juang, A.T. Huang, Mater. Lett. 137, 136 (2014)<br>
[3] 	Liang, Y. C., Chih Chen, and King-Ning Tu. "Side wall wetting induced void formation due to small solder volume in microbumps of Ni/SnAg/Ni upon reflow." ECS Solid State Letters 1.4 (2012): P60.<br>
[4] 	LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.<br>
[5] 	Vaswani, Ashish, et al. "Attention is all you need." arXiv preprint arXiv:1706.03762 (2017).<br>
[6] 	S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Müller, and W. Samek, “On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation,” PLOS ONE, vol. 10, no. 7, p. e0130140, Jul. 2015.<br>
[7] 	H. K. Kim, H. K. Liou, and K. N. Tu, J. Materials Research, 10, 497(1995)<br>
[8] 	J.S. Hwang and R.M. Vargas, Soldering & Surface Mount Technology, 2, 38 (1990)<br>
[9] 	V.B. Fiks, USoviet Physics-Solid StateU, 1,pp.14-28,1959.<br>
[10] 	J. W. Jang, D. R. Frear, T. Y. Lee and K. N. Tu, J. Applied Physics, 88, 6359 (2000)<br>
[11] 	Y.C. Liang, C. Chen, K.N. Tu, ECS Solid State Lett. 1, 60 (2012).<br>
[12] 	Chen, Chih, Doug Yu, and Kuan-Neng Chen. "Vertical interconnects of microbumps in 3D integration." MRS bulletin 40.3 (2015): 257-263.<br>
[13] 	Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012).<br>
[14] 	Huang, Xiaojie, Junjie Shan, and Vivek Vaidya. "Lung nodule detection in CT using 3D convolutional neural networks." 2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017). IEEE, 2017.<br>
[15] 	Gao, Xiaohong W., Rui Hui, and Zengmin Tian. "Classification of CT brain images based on deep learning networks." Computer methods and programs in biomedicine 138 (2017): 49-56.<br>
[16] 	Gonzalez, Rafael C.; Woods, Richard E. (2008). Digital Image Processing (3rd ed.). Prentice Hall. p. 128.(3.1)<br>
[17] 	Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Masayoshi Tomizuka, Kurt Keutzer, and Peter Vajda. Visual transformers: Token-based image representation and processing for computer vision. arXiv:2006.03677, 2020.
[18] 	Woo, Sanghyun, et al. "Cbam: Convolutional block attention module." Proceedings of the European conference on computer vision (ECCV). 2018.
[19] 	Samek, Wojciech, et al., eds. Explainable AI: interpreting, explaining and visualizing deep learning. Vol. 11700. Springer Nature, 2019.



















