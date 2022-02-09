# Resistance Prediction  

## 網路結構  
<font size=3>主要由兩層conv.再加上一個VT module組成。前端的conv.負責萃取特徵並降低資料維度，輸出的shape為:<br>
`(channels, depth, height, width) -> (32, 16, 16, 16)`<br>
    
VT module中的tokenizer則會先將feature map攤平，並透過Spacial Attention將feature map拆分為指定的token數量，最後再將token的維度壓縮到指定的channel數量，。
```
# flatten
feature_map(c, d, h, w) to (d*h*w, c) -> (4096, 32)
# spacial attention
wA = nn.Linear(c, num_tokens) -> (4096, 32)*(32, 8) = (4096, 8)
T = transpose(wA)*feature_map -> (8, 4096)*(4096, 32) = (8, 32)
T = nn.Linear(c, token_dim) -> (8, 32)*(32, 32) = (8, 32)
```
</font>  
<center><img src="https://i.imgur.com/0ojwdxN.png" width="50%" alt="img01"/></center>  

