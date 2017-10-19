# 這是我的論文程式環境

>詳細操作請看 程式文件-aug-26-2017.pdf
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


>論文理論則看 論文初稿20170828.pdf
