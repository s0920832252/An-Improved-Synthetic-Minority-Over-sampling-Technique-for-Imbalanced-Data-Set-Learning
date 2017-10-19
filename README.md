# 這是我的論文程式環境

>詳細操作請看 程式文件-aug-26-2017.pdf
>> 預設的實驗流程是
1. over-sampling 方法替訓練集產生人造實例後得到一個新的訓練集，此新的
訓練集會各自使用 KNN/DT/SVM/NB/Logitc 等五種分類演算法訓練一個分
類器(共五個)，此五個分類器會以 Accuracy/TP/FP/Precision/AUC/Gmean/F-measure
七種效能指標衡量分類器的表現。
2. 上述動作每一個 over-sampling 方法都會執行一次
3. 預設要一起比較的 over-sampling 方法為
{ISMOTE,B-SMOTE,ADASYN,MSMOTE,MWMOTE,SMOTE}共六種
ISMOTE 方法在使用 Script_Experience 函數時可替換成別的方法。
>論文理論則看 論文初稿20170828.pdf
