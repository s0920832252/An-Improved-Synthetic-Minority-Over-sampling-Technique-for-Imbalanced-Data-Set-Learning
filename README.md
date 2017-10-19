# 這是我的論文程式環境

>[詳細操作請看 程式文件-aug-26-2017.pdf](https://github.com/s0920832252/An-Improved-Synthetic-Minority-Over-sampling-Technique-for-Imbalanced-Data-Set-Learning/blob/master/%E7%A8%8B%E5%BC%8F%E6%96%87%E4%BB%B6-aug-26-2017.pdf)
>> 預設的實驗流程是
>> 1. over-sampling 方法替訓練集產生人造實例後得到一個新的訓練集，此新的
>> 訓練集會各自使用 KNN/DT/SVM/NB/Logitc 等五種分類演算法訓練一個分類器(共五個)，此五個分類器會以 Accuracy/TP/FP/Precision/AUC/Gmean/F-measure
>> 七種效能指標衡量分類器的表現。
>> 2. 上述動作每一個 over-sampling 方法都會執行一次
>> 3. 預設要一起比較的 over-sampling 方法為{ISMOTE,B-SMOTE,ADASYN,MSMOTE,MWMOTE,SMOTE}共六種
>> ISMOTE 方法在使用 Script_Experience 函數時可替換成別的方法。
>> 
>> Script_Experience 
>> 1. 使用者可透過此函數設計出自己的實驗流程
>> 2. Script_Experience 的設計概念為實驗的腳本，使用者可一次安排複數的實驗檢定。，亦可直接在 Matlab 視窗下呼叫
>> Performance_Wilcoxon 或者 Performance_t_test 程式在執行完成後輸出該程式執行的結果。 
>>


>[論文理論則看 論文初稿20170828.pdf](https://github.com/s0920832252/An-Improved-Synthetic-Minority-Over-sampling-Technique-for-Imbalanced-Data-Set-Learning/blob/master/%E8%AB%96%E6%96%87%E5%88%9D%E7%A8%BF20170828.pdf)
>>摘要
>>> 當資料集的少數類別實例有相對其他類別較少的實例數目時，則這樣的資料集可能隱含著類別不平衡的問題，也就是說訓練出的分類模型很可能因為少數類別實例發現機率較低的原因，而將少數類別實例錯誤判斷成多數類別實例。透過人造少數類別資料實例以平衡多數類別以及少數類別之間的分佈不平衡是一種解決策略。有多種演算法已經依據此概念被設計出來。本研究提出一個改良的演算法 ISMOTE 來解決類別不平衡問題。ISMOTE 與以往演算法不同的地方是並非僅考慮少數類別的分布，而是同時衡量少數類別和多數類別在密度分布上的相對優勢，並以此作為權重衡量的基礎。另外，我們的方法會選擇以少數類別實例與距離其最近的多數類別實例作為參考實例產生人造實例。此作法可減少因為產生錯誤的人造資料實例而使分類器的學習更加地困難的狀況發生，並且透過此作法的人造實例能更好的幫助分類器的學習。每一個少數類別實例具有一個分類器對於此資料實例困難學習的權重。權重公式的設計原則與此少數類別資料實例的困難學習程度呈成正比。 因此ISMOTE 可以針對每一個少數類別資料實例的權重，產生相對應數量的人造資料實例而逐漸改變分類決策的界線往較困難學習的方向。

