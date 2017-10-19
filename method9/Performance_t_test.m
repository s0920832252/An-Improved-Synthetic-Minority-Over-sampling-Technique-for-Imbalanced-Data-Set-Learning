function [  ] = Performance_t_test( yourFunction,yourFunctionName,outTextFileName )

%  dbstop if error %如果有問題matlab會停在出錯的那行，並且保存所有相關變數-除錯用

tic  %碼表開始倒數

%使用的訓練集
trainFile={'abalone9-18.mat','Btissue.mat','ecoli1.mat'...
    ,'pima.mat','vehicle1.mat'...
    ,'yeast.mat','segment0.mat'...
    ,'wisconsin.mat','appendicitis.mat','bupa.mat','heart.mat'...
    ,'haberman.mat','glass0.mat','new-thyroid1.mat','ecoli3.mat','vehicle3.mat','wdbc.mat'...
    ,'cleveland-0_vs_4.mat','page-blocks0.mat','Robot.mat'};

%%設定資料集所在根目錄
path='./data/';
%%設定K-FOLD次數
KFold=5;
%%設定鄰居數
KNighbor=5;
%做test n次
t_testCount=5;
%實驗方法名稱-輸出用
method={yourFunctionName,'B-SMOTE','ADAYSN','MSMOTE','MWMOTE','SMOTE'};
%檢定的阿法值
alpha=0.1; 
%方法數
methodCount=size(method,2);;



%參數初始化
%KNN totoal index
KNN_Accurecy=zeros(methodCount,t_testCount);
KNN_TP=zeros(methodCount,t_testCount);
KNN_FP=zeros(methodCount,t_testCount);
KNN_Precision=zeros(methodCount,t_testCount);
KNN_AUC=zeros(methodCount,t_testCount);
KNN_G_mean=zeros(methodCount,t_testCount);
KNN_F_measure=zeros(methodCount,t_testCount);

%DT totoal index
DT_Accurecy=zeros(methodCount,t_testCount);
DT_TP=zeros(methodCount,t_testCount);
DT_FP=zeros(methodCount,t_testCount);
DT_Precision=zeros(methodCount,t_testCount);
DT_AUC=zeros(methodCount,t_testCount);
DT_G_mean=zeros(methodCount,t_testCount);
DT_F_measure=zeros(methodCount,t_testCount);

%SVM totoal index
SVM_Accurecy=zeros(methodCount,t_testCount);
SVM_TP=zeros(methodCount,t_testCount);
SVM_FP=zeros(methodCount,t_testCount);
SVM_Precision=zeros(methodCount,t_testCount);
SVM_AUC=zeros(methodCount,t_testCount);
SVM_G_mean=zeros(methodCount,t_testCount);
SVM_F_measure=zeros(methodCount,t_testCount);

%NaiveB totoal index
NaiveB_Accurecy=zeros(methodCount,t_testCount);
NaiveB_TP=zeros(methodCount,t_testCount);
NaiveB_FP=zeros(methodCount,t_testCount);
NaiveB_Precision=zeros(methodCount,t_testCount);
NaiveB_AUC=zeros(methodCount,t_testCount);
NaiveB_G_mean=zeros(methodCount,t_testCount);
NaiveB_F_measure=zeros(methodCount,t_testCount);

%logisticR totoal index
logisticR_Accurecy=zeros(methodCount,t_testCount);
logisticR_TP=zeros(methodCount,t_testCount);
logisticR_FP=zeros(methodCount,t_testCount);
logisticR_Precision=zeros(methodCount,t_testCount);
logisticR_AUC=zeros(methodCount,t_testCount);
logisticR_G_mean=zeros(methodCount,t_testCount);
logisticR_F_measure=zeros(methodCount,t_testCount);


for fileIndex=1 : size(trainFile,2)
    
    %讀取資料
    totalData=load([path  trainFile{fileIndex}]);
    totalData=totalData.X;
        
    %下列的for - 刪除變異數太小的feature  
    delNo=[];
    spiltMinorCount=floor(size(find(totalData(:,size(totalData,2))==1),1)/KFold);
    spiltMajorCount=floor((size(totalData,1)-size(find(totalData(:,size(totalData,2))==1),1))/KFold);           
    for kfoldIndex=1 : KFold
        testMajorDataNo=1+(kfoldIndex-1)*spiltMajorCount:kfoldIndex*spiltMajorCount;
        testMinorDataNo=1+(kfoldIndex-1)*spiltMinorCount:kfoldIndex*spiltMinorCount;
        minorClassInstanceNo=find(totalData(:,size(totalData,2))==1);
        majorClassInstanceNo=setdiff(1:size(totalData,1),minorClassInstanceNo);
        testDataNo=union(minorClassInstanceNo(testMinorDataNo),majorClassInstanceNo(testMajorDataNo));
        trainDataNo=setdiff(1:size(totalData,1),testDataNo);
        testData=totalData(testDataNo,:);
        trainData=totalData(trainDataNo,:);
          
        for i=1:size(totalData,2)-1
            if var(trainData(trainData(:,size(totalData,2))==1,i))<=0.0001
                delNo=union(delNo,i);
            end
        end
    end
    totalData(:,delNo)=[];
    
    %得到此訓練集相關資料
    instanceNum=size(totalData,1);
    featureNum=size(totalData,2);
    [majorClassNo,minorClassNo]=recognizeMajorClassAndOtherClass(totalData);
    minorInstanceNum=size(find(totalData(:,featureNum)==minorClassNo),1);
    majorInstanceNum=instanceNum-minorInstanceNum;
    classRatio=majorInstanceNum/minorInstanceNum;
    
    %共作t_testCount次 最後一次才顯示結果
    again=true;
    count=0;
    while again
        
        %KNN計算指標初始化
        ACK=zeros(methodCount,KFold);
        TK=zeros(methodCount,KFold);
        FPK=zeros(methodCount,KFold);
        PK=zeros(methodCount,KFold);
        AK=zeros(methodCount,KFold);
        GK=zeros(methodCount,KFold);
        FK=zeros(methodCount,KFold);
        
        %     decisionTree 計算指標初始化
        ACT=zeros(methodCount,KFold);
        TT=zeros(methodCount,KFold);
        FPT=zeros(methodCount,KFold);
        PT=zeros(methodCount,KFold);
        AT=zeros(methodCount,KFold);
        GT=zeros(methodCount,KFold);
        FT=zeros(methodCount,KFold);
                
        %     SVM 計算指標初始化
        ACS=zeros(methodCount,KFold);
        TS=zeros(methodCount,KFold);
        FPS=zeros(methodCount,KFold);
        PS=zeros(methodCount,KFold);
        AS=zeros(methodCount,KFold);
        GS=zeros(methodCount,KFold);
        FS=zeros(methodCount,KFold);
        
        %     NAVIE 計算指標初始化
        ACN=zeros(methodCount,KFold);
        TN=zeros(methodCount,KFold);
        FPN=zeros(methodCount,KFold);
        PN=zeros(methodCount,KFold);
        AN=zeros(methodCount,KFold);
        GN=zeros(methodCount,KFold);
        FN=zeros(methodCount,KFold);
        
        %     logisit 計算指標初始化
        ACL=zeros(methodCount,KFold);
        TL=zeros(methodCount,KFold);
        FPL=zeros(methodCount,KFold);
        PL=zeros(methodCount,KFold);
        AL=zeros(methodCount,KFold);
        GL=zeros(methodCount,KFold);
        FL=zeros(methodCount,KFold);
        
        %每一個Fold的資料數量
        spiltMinorCount=floor(minorInstanceNum/KFold);
        spiltMajorCount=floor(majorInstanceNum/KFold);
        
        %執行Kfold
        for kfoldIndex=1 : KFold
            %少數類別資料和多數資料分別切成 KFold分   依據FOld數將少數以及多數組合成一個Fold  (避免直接切成K-fold 有些FOLD沒有少數類別)
            disp([trainFile{fileIndex} num2str(kfoldIndex) '-' num2str(count)]);
            testMinorDataNo=1+(kfoldIndex-1)*spiltMinorCount:kfoldIndex*spiltMinorCount;
            testMajorDataNo=1+(kfoldIndex-1)*spiltMajorCount:kfoldIndex*spiltMajorCount;
            minorClassInstanceNo=find(totalData(:,featureNum)==minorClassNo);
            majorClassInstanceNo=setdiff(1:instanceNum,minorClassInstanceNo);  %多數類別資料在全部資料內的編號
            
            %得到testSet以及trainSet
            testDataNo=union(minorClassInstanceNo(testMinorDataNo),majorClassInstanceNo(testMajorDataNo));
            trainDataNo=setdiff(1:instanceNum,testDataNo);
            testData=totalData(testDataNo,:);
            trainData=totalData(trainDataNo,:);
            
%             得到實驗設定的參數
            minorClassSetSize=size(find(trainData(:,featureNum)==minorClassNo),1); %少數類別的數量
            majorClassSetSize=size(trainDataNo,2)-minorClassSetSize;
            makeAmount=majorClassSetSize-minorClassSetSize;
            radio=round(majorClassSetSize/minorClassSetSize);
            
            %執行match個方法以及在五個環境
            for match=1 : methodCount                
                switch match                    
                    case {1}
                        yourDataSet=yourFunction(trainData,majorClassNo,minorClassNo,KNighbor);
                        [tempknnAccurecy,tempknnTPK,tempknnFP,tempknnPrecision,tempknnAUC,tempknnG_mean,tempknnF_measure]=KnnA(yourDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempTreeAccurecy,tempTreeTP,tempTreeFP,tempTreePrecision,tempTreeAUC,tempTreeG_mean,tempTreeF_measure]=decTree(yourDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempSVMAccurecy,tempSVMTPK,tempSVMFP,tempSVMPrecision,tempSVMAUC,tempSVMG_mean,tempSVMF_measure]=Svm(yourDataSet,testData,majorClassNo,minorClassNo,featureNum);                        
                        [tempNBAccurecy,tempNBTP,tempNBFP,tempNBPrecision,tempNBAUC,tempNBG_mean,tempNBF_measure]=naiveBayes(yourDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempLogRAccurecy,tempLogRTP,tempLogRFP,tempLogRPrecision,tempLogRAUC,tempLogRG_mean,tempLogRF_measure]=logistic(yourDataSet,testData,majorClassNo,minorClassNo,featureNum);
                                
                    case {2}
                        Borderline_SMOTEDataSet=Borderline_SMOTE(trainData,minorClassNo,KNighbor);
                        [tempknnAccurecy,tempknnTPK,tempknnFP,tempknnPrecision,tempknnAUC,tempknnG_mean,tempknnF_measure]=KnnA(Borderline_SMOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempTreeAccurecy,tempTreeTP,tempTreeFP,tempTreePrecision,tempTreeAUC,tempTreeG_mean,tempTreeF_measure]=decTree(Borderline_SMOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempSVMAccurecy,tempSVMTPK,tempSVMFP,tempSVMPrecision,tempSVMAUC,tempSVMG_mean,tempSVMF_measure]=Svm(Borderline_SMOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);                        
                        [tempNBAccurecy,tempNBTP,tempNBFP,tempNBPrecision,tempNBAUC,tempNBG_mean,tempNBF_measure]=naiveBayes(Borderline_SMOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempLogRAccurecy,tempLogRTP,tempLogRFP,tempLogRPrecision,tempLogRAUC,tempLogRG_mean,tempLogRF_measure]=logistic(Borderline_SMOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        
                    case {3}
                        ADAYSNDataSet=ADAYSN(trainData,minorClassNo,KNighbor);
                        [tempknnAccurecy,tempknnTPK,tempknnFP,tempknnPrecision,tempknnAUC,tempknnG_mean,tempknnF_measure]=KnnA(ADAYSNDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempTreeAccurecy,tempTreeTP,tempTreeFP,tempTreePrecision,tempTreeAUC,tempTreeG_mean,tempTreeF_measure]=decTree(ADAYSNDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempSVMAccurecy,tempSVMTPK,tempSVMFP,tempSVMPrecision,tempSVMAUC,tempSVMG_mean,tempSVMF_measure]=Svm(ADAYSNDataSet,testData,majorClassNo,minorClassNo,featureNum);                        
                        [tempNBAccurecy,tempNBTP,tempNBFP,tempNBPrecision,tempNBAUC,tempNBG_mean,tempNBF_measure]=naiveBayes(ADAYSNDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempLogRAccurecy,tempLogRTP,tempLogRFP,tempLogRPrecision,tempLogRAUC,tempLogRG_mean,tempLogRF_measure]=logistic(ADAYSNDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        
                    case {4}
                        MSNOTEDataSet=MSNOTE(trainData,minorClassNo,radio,KNighbor);
                        [tempknnAccurecy,tempknnTPK,tempknnFP,tempknnPrecision,tempknnAUC,tempknnG_mean,tempknnF_measure]=KnnA(MSNOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempTreeAccurecy,tempTreeTP,tempTreeFP,tempTreePrecision,tempTreeAUC,tempTreeG_mean,tempTreeF_measure]=decTree(MSNOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempSVMAccurecy,tempSVMTPK,tempSVMFP,tempSVMPrecision,tempSVMAUC,tempSVMG_mean,tempSVMF_measure]=Svm(MSNOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);                        
                        [tempNBAccurecy,tempNBTP,tempNBFP,tempNBPrecision,tempNBAUC,tempNBG_mean,tempNBF_measure]=naiveBayes(MSNOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempLogRAccurecy,tempLogRTP,tempLogRFP,tempLogRPrecision,tempLogRAUC,tempLogRG_mean,tempLogRF_measure]=logistic(MSNOTEDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        
                    case {5}
                        MWMOTEDateSet=MWMOTE(trainData,minorClassNo,makeAmount,KNighbor,3,3);
                        [tempknnAccurecy,tempknnTPK,tempknnFP,tempknnPrecision,tempknnAUC,tempknnG_mean,tempknnF_measure]=KnnA(MWMOTEDateSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempTreeAccurecy,tempTreeTP,tempTreeFP,tempTreePrecision,tempTreeAUC,tempTreeG_mean,tempTreeF_measure]=decTree(MWMOTEDateSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempSVMAccurecy,tempSVMTPK,tempSVMFP,tempSVMPrecision,tempSVMAUC,tempSVMG_mean,tempSVMF_measure]=Svm(MWMOTEDateSet,testData,majorClassNo,minorClassNo,featureNum);                        
                        [tempNBAccurecy,tempNBTP,tempNBFP,tempNBPrecision,tempNBAUC,tempNBG_mean,tempNBF_measure]=naiveBayes(MWMOTEDateSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempLogRAccurecy,tempLogRTP,tempLogRFP,tempLogRPrecision,tempLogRAUC,tempLogRG_mean,tempLogRF_measure]=logistic(MWMOTEDateSet,testData,majorClassNo,minorClassNo,featureNum);
                        
                    case {6}
                        SmoteDataSet=Smote(trainData,minorClassNo,radio,KNighbor);
                        [tempknnAccurecy,tempknnTPK,tempknnFP,tempknnPrecision,tempknnAUC,tempknnG_mean,tempknnF_measure]=KnnA(SmoteDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempTreeAccurecy,tempTreeTP,tempTreeFP,tempTreePrecision,tempTreeAUC,tempTreeG_mean,tempTreeF_measure]=decTree(SmoteDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempSVMAccurecy,tempSVMTPK,tempSVMFP,tempSVMPrecision,tempSVMAUC,tempSVMG_mean,tempSVMF_measure]=Svm(SmoteDataSet,testData,majorClassNo,minorClassNo,featureNum);                        
                        [tempNBAccurecy,tempNBTP,tempNBFP,tempNBPrecision,tempNBAUC,tempNBG_mean,tempNBF_measure]=naiveBayes(SmoteDataSet,testData,majorClassNo,minorClassNo,featureNum);
                        [tempLogRAccurecy,tempLogRTP,tempLogRFP,tempLogRPrecision,tempLogRAUC,tempLogRG_mean,tempLogRF_measure]=logistic(SmoteDataSet,testData,majorClassNo,minorClassNo,featureNum);
                end
                
                %     KNN 計算指標
                ACK(match,kfoldIndex)=tempknnAccurecy;
                TK(match,kfoldIndex)=tempknnTPK;
                FPK(match,kfoldIndex)=tempknnFP;
                PK(match,kfoldIndex)=tempknnPrecision;
                AK(match,kfoldIndex)=tempknnAUC;
                GK(match,kfoldIndex)=tempknnG_mean;
                FK(match,kfoldIndex)=tempknnF_measure;
                
                %     DT 計算指標
                ACT(match,kfoldIndex)=tempTreeAccurecy;
                TT(match,kfoldIndex)=tempTreeTP;
                FPT(match,kfoldIndex)=tempTreeFP;
                PT(match,kfoldIndex)=tempTreePrecision;
                AT(match,kfoldIndex)=tempTreeAUC;
                GT(match,kfoldIndex)=tempTreeG_mean;
                FT(match,kfoldIndex)=tempTreeF_measure;
                
                %     SVM 計算指標
                ACS(match,kfoldIndex)=tempSVMAccurecy;
                TS(match,kfoldIndex)=tempSVMTPK;
                FPS(match,kfoldIndex)=tempSVMFP;
                PS(match,kfoldIndex)=tempSVMPrecision;
                AS(match,kfoldIndex)=tempSVMAUC;
                GS(match,kfoldIndex)=tempSVMG_mean;
                FS(match,kfoldIndex)=tempSVMF_measure;
                
                %     NAVIE 計算指標
                ACN(match,kfoldIndex)=tempNBAccurecy;
                TN(match,kfoldIndex)=tempNBTP;
                FPN(match,kfoldIndex)=tempNBFP;
                PN(match,kfoldIndex)=tempNBPrecision;
                AN(match,kfoldIndex)=tempNBAUC;
                GN(match,kfoldIndex)=tempNBG_mean;
                FN(match,kfoldIndex)=tempNBF_measure;
                
                %     logisit 計算指標
                ACL(match,kfoldIndex)=tempLogRAccurecy;
                TL(match,kfoldIndex)=tempLogRTP;
                FPL(match,kfoldIndex)=tempLogRFP;
                PL(match,kfoldIndex)=tempLogRPrecision;
                AL(match,kfoldIndex)=tempLogRAUC;
                GL(match,kfoldIndex)=tempLogRG_mean;
                FL(match,kfoldIndex)=tempLogRF_measure;
            end
        end
        
        %最後一次t-t-est次數  要顯示結果囉~
        count=count+1;
        if count==t_testCount
            again=false;
        end
        
        %求出第幾次的K-fold平均值
        for i=1:methodCount
            KNN_Accurecy(i,count)=mean(ACK(i,:));
            DT_Accurecy(i,count)=mean(ACT(i,:));
            SVM_Accurecy(i,count)=mean(ACS(i,:));
            NaiveB_Accurecy(i,count)=mean(ACN(i,:));
            logisticR_Accurecy(i,count)=mean(ACL(i,:));
            
            KNN_TP(i,count)=mean(TK(i,:));
            DT_TP(i,count)=mean(TT(i,:));
            SVM_TP(i,count)=mean(TS(i,:));
            NaiveB_TP(i,count)=mean(TN(i,:));
            logisticR_TP(i,count)=mean(TL(i,:));
            
            KNN_FP(i,count)=mean(FPK(i,:));
            DT_FP(i,count)=mean(FPT(i,:));
            SVM_FP(i,count)=mean(FPS(i,:));
            NaiveB_FP(i,count)=mean(FPN(i,:));
            logisticR_FP(i,count)=mean(FPL(i,:));
            
            KNN_Precision(i,count)=mean(PK(i,:));
            DT_Precision(i,count)=mean(PT(i,:));
            SVM_Precision(i,count)=mean(PS(i,:));
            NaiveB_Precision(i,count)=mean(PN(i,:));
            logisticR_Precision(i,count)=mean(PL(i,:));
            
            KNN_AUC(i,count)=mean(AK(i,:));
            DT_AUC(i,count)=mean(AT(i,:));
            SVM_AUC(i,count)=mean(AS(i,:));
            NaiveB_AUC(i,count)=mean(AN(i,:));
            logisticR_AUC(i,count)=mean(AL(i,:));
            
            KNN_G_mean(i,count)=mean(GK(i,:));
            DT_G_mean(i,count)=mean(GT(i,:));
            SVM_G_mean(i,count)=mean(GS(i,:));
            NaiveB_G_mean(i,count)=mean(GN(i,:));
            logisticR_G_mean(i,count)=mean(GL(i,:));
            
            KNN_F_measure(i,count)=mean(FK(i,:));
            DT_F_measure(i,count)=mean(FT(i,:));
            SVM_F_measure(i,count)=mean(FS(i,:));
            NaiveB_F_measure(i,count)=mean(FN(i,:));
            logisticR_F_measure(i,count)=mean(FL(i,:));
        end
                
        if ~again                     
            indexName={'Accurecy','TP','FP','Precision','AUC','G-mean','F-measure'};  %使用index名稱
            
            %開檔
            fileID=fopen(strcat(trainFile{fileIndex},'-',outTextFileName,"_t-test檢定.txt") ,'w');
            
            %計算各個指標的 檢定結果 P-Value 信賴區間(未使用) 相關數值(僅使用標準差)
            for index_choice=1:7
                for Competitor_choice=2:methodCount
                    switch index_choice
                        case {1}   %                 Accurecy
                            [h1,p1,ci1,stats1] =ttest(KNN_Accurecy(1,:),KNN_Accurecy(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h2,p2,ci2,stats2] =ttest(DT_Accurecy(1,:),DT_Accurecy(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h3,p3,ci3,stats3] =ttest(SVM_Accurecy(1,:),SVM_Accurecy(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h4,p4,ci4,stats4] =ttest(NaiveB_Accurecy(1,:),NaiveB_Accurecy(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h5,p5,ci5,stats5] =ttest(logisticR_Accurecy(1,:),logisticR_Accurecy(Competitor_choice,:),'tail','right','alpha', alpha);
                            
                            [h11,p11,ci11,stats11] =ttest(KNN_Accurecy(1,:),KNN_Accurecy(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h22,p22,ci22,stats22] =ttest(DT_Accurecy(1,:),DT_Accurecy(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h33,p33,ci33,stats33] =ttest(SVM_Accurecy(1,:),SVM_Accurecy(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h44,p44,ci44,stats44] =ttest(NaiveB_Accurecy(1,:),NaiveB_Accurecy(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h55,p55,ci55,stats55] =ttest(logisticR_Accurecy(1,:),logisticR_Accurecy(Competitor_choice,:),'tail','left','alpha', alpha);
                            
                            mean1=mean(KNN_Accurecy(1,:)-KNN_Accurecy(Competitor_choice,:));
                            std1=stats1.sd;
                            mean2=mean(DT_Accurecy(1,:)-DT_Accurecy(Competitor_choice,:));
                            std2=stats2.sd;
                            mean3=mean(SVM_Accurecy(1,:)-SVM_Accurecy(Competitor_choice,:));
                            std3=stats3.sd;
                            mean4=mean(NaiveB_Accurecy(1,:)-NaiveB_Accurecy(Competitor_choice,:));
                            std4=stats4.sd;
                            mean5=mean(logisticR_Accurecy(1,:)-logisticR_Accurecy(Competitor_choice,:));
                            std5=stats5.sd;
                            
                        case {2}%                 TP
                            [h1,p1,ci1,stats1] =ttest(KNN_TP(1,:),KNN_TP(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h2,p2,ci2,stats2] =ttest(DT_TP(1,:),DT_TP(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h3,p3,ci3,stats3] =ttest(SVM_TP(1,:),SVM_TP(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h4,p4,ci4,stats4] =ttest(NaiveB_TP(1,:),NaiveB_TP(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h5,p5,ci5,stats5] =ttest(logisticR_TP(1,:),logisticR_TP(Competitor_choice,:),'tail','right','alpha', alpha);
                            
                            [h11,p11,ci11,stats11] =ttest(KNN_TP(1,:),KNN_TP(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h22,p22,ci22,stats22] =ttest(DT_TP(1,:),DT_TP(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h33,p33,ci33,stats33] =ttest(SVM_TP(1,:),SVM_TP(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h44,p44,ci44,stats44] =ttest(NaiveB_TP(1,:),NaiveB_TP(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h55,p55,ci55,stats55] =ttest(logisticR_TP(1,:),logisticR_TP(Competitor_choice,:),'tail','left','alpha', alpha);
                            
                            mean1=mean(KNN_TP(1,:)-KNN_TP(Competitor_choice,:));
                            std1=stats1.sd;
                            mean2=mean(DT_TP(1,:)-DT_TP(Competitor_choice,:));
                            std2=stats2.sd;
                            mean3=mean(SVM_TP(1,:)-SVM_TP(Competitor_choice,:));
                            std3=stats3.sd;
                            mean4=mean(NaiveB_TP(1,:)-NaiveB_TP(Competitor_choice,:));
                            std4=stats4.sd;
                            mean5=mean(logisticR_TP(1,:)-logisticR_TP(Competitor_choice,:));
                            std5=stats5.sd;
                        case {3} %                 FP
                            [h1,p1,ci1,stats1] =ttest(KNN_FP(1,:),KNN_FP(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h2,p2,ci2,stats2] =ttest(DT_FP(1,:),DT_FP(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h3,p3,ci3,stats3] =ttest(SVM_FP(1,:),SVM_FP(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h4,p4,ci4,stats4] =ttest(NaiveB_FP(1,:),NaiveB_FP(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h5,p5,ci5,stats5] =ttest(logisticR_FP(1,:),logisticR_FP(Competitor_choice,:),'tail','left','alpha', alpha);
                            
                            [h11,p11,ci11,stats11] =ttest(KNN_FP(1,:),KNN_FP(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h22,p22,ci22,stats22] =ttest(DT_FP(1,:),DT_FP(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h33,p33,ci33,stats33] =ttest(SVM_FP(1,:),SVM_FP(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h44,p44,ci44,stats44] =ttest(NaiveB_FP(1,:),NaiveB_FP(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h55,p55,ci55,stats55] =ttest(logisticR_FP(1,:),logisticR_FP(Competitor_choice,:),'tail','right','alpha', alpha);
                            
                            mean1=mean(KNN_FP(1,:)-KNN_FP(Competitor_choice,:));
                            std1=stats1.sd;
                            mean2=mean(DT_FP(1,:)-DT_FP(Competitor_choice,:));
                            std2=stats2.sd;
                            mean3=mean(SVM_FP(1,:)-SVM_FP(Competitor_choice,:));
                            std3=stats3.sd;
                            mean4=mean(NaiveB_FP(1,:)-NaiveB_FP(Competitor_choice,:));
                            std4=stats4.sd;
                            mean5=mean(logisticR_FP(1,:)-logisticR_FP(Competitor_choice,:));
                            std5=stats5.sd;
                        case {4}%                 Precision
                            [h1,p1,ci1,stats1] =ttest(KNN_Precision(1,:),KNN_Precision(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h2,p2,ci2,stats2] =ttest(DT_Precision(1,:),DT_Precision(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h3,p3,ci3,stats3] =ttest(SVM_Precision(1,:),SVM_Precision(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h4,p4,ci4,stats4] =ttest(NaiveB_Precision(1,:),NaiveB_Precision(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h5,p5,ci5,stats5] =ttest(logisticR_Precision(1,:),logisticR_Precision(Competitor_choice,:),'tail','right','alpha', alpha);
                            
                            [h11,p11,ci11,stats11] =ttest(KNN_Precision(1,:),KNN_Precision(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h22,p22,ci22,stats22] =ttest(DT_Precision(1,:),DT_Precision(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h33,p33,ci33,stats33] =ttest(SVM_Precision(1,:),SVM_Precision(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h44,p44,ci44,stats44] =ttest(NaiveB_Precision(1,:),NaiveB_Precision(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h55,p55,ci55,stats55] =ttest(logisticR_Precision(1,:),logisticR_Precision(Competitor_choice,:),'tail','left','alpha', alpha);
                            
                            mean1=mean(KNN_Precision(1,:)-KNN_Precision(Competitor_choice,:));
                            std1=stats1.sd;
                            mean2=mean(DT_Precision(1,:)-DT_Precision(Competitor_choice,:));
                            std2=stats2.sd;
                            mean3=mean(SVM_Precision(1,:)-SVM_Precision(Competitor_choice,:));
                            std3=stats3.sd;
                            mean4=mean(NaiveB_Precision(1,:)-NaiveB_Precision(Competitor_choice,:));
                            std4=stats4.sd;
                            mean5=mean(logisticR_Precision(1,:)-logisticR_Precision(Competitor_choice,:));
                            std5=stats5.sd;
                            
                        case {5}%                 AUC
                            [h1,p1,ci1,stats1] =ttest(KNN_AUC(1,:),KNN_AUC(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h2,p2,ci2,stats2] =ttest(DT_AUC(1,:),DT_AUC(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h3,p3,ci3,stats3] =ttest(SVM_AUC(1,:),SVM_AUC(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h4,p4,ci4,stats4] =ttest(NaiveB_AUC(1,:),NaiveB_AUC(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h5,p5,ci5,stats5] =ttest(logisticR_AUC(1,:),logisticR_AUC(Competitor_choice,:),'tail','right','alpha', alpha);
                            
                            [h11,p11,ci11,stats11] =ttest(KNN_AUC(1,:),KNN_AUC(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h22,p22,ci22,stats22] =ttest(DT_AUC(1,:),DT_AUC(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h33,p33,ci33,stats33] =ttest(SVM_AUC(1,:),SVM_AUC(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h44,p44,ci44,stats44] =ttest(NaiveB_AUC(1,:),NaiveB_AUC(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h55,p55,ci55,stats55] =ttest(logisticR_AUC(1,:),logisticR_AUC(Competitor_choice,:),'tail','left','alpha', alpha);
                            
                            mean1=mean(KNN_AUC(1,:)-KNN_AUC(Competitor_choice,:));
                            std1=stats1.sd;
                            mean2=mean(DT_AUC(1,:)-DT_AUC(Competitor_choice,:));
                            std2=stats2.sd;
                            mean3=mean(SVM_AUC(1,:)-SVM_AUC(Competitor_choice,:));
                            std3=stats3.sd;
                            mean4=mean(NaiveB_AUC(1,:)-NaiveB_AUC(Competitor_choice,:));
                            std4=stats4.sd;
                            mean5=mean(logisticR_AUC(1,:)-logisticR_AUC(Competitor_choice,:));
                            std5=stats5.sd;
                        case {6}%                 G_mean
                            [h1,p1,ci1,stats1] =ttest(KNN_G_mean(1,:),KNN_G_mean(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h2,p2,ci2,stats2] =ttest(DT_G_mean(1,:),DT_G_mean(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h3,p3,ci3,stats3] =ttest(SVM_G_mean(1,:),SVM_G_mean(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h4,p4,ci4,stats4] =ttest(NaiveB_G_mean(1,:),NaiveB_G_mean(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h5,p5,ci5,stats5] =ttest(logisticR_G_mean(1,:),logisticR_G_mean(Competitor_choice,:),'tail','right','alpha', alpha);
                            
                            [h11,p11,ci11,stats11] =ttest(KNN_G_mean(1,:),KNN_G_mean(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h22,p22,ci22,stats22] =ttest(DT_G_mean(1,:),DT_G_mean(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h33,p33,ci33,stats33] =ttest(SVM_G_mean(1,:),SVM_G_mean(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h44,p44,ci44,stats44] =ttest(NaiveB_G_mean(1,:),NaiveB_G_mean(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h55,p55,ci55,stats55] =ttest(logisticR_G_mean(1,:),logisticR_G_mean(Competitor_choice,:),'tail','left','alpha', alpha);
                            
                            mean1=mean(KNN_G_mean(1,:)-KNN_G_mean(Competitor_choice,:));
                            std1=stats1.sd;
                            mean2=mean(DT_G_mean(1,:)-DT_G_mean(Competitor_choice,:));
                            std2=stats2.sd;
                            mean3=mean(SVM_G_mean(1,:)-SVM_G_mean(Competitor_choice,:));
                            std3=stats3.sd;
                            mean4=mean(NaiveB_G_mean(1,:)-NaiveB_G_mean(Competitor_choice,:));
                            std4=stats4.sd;
                            mean5=mean(logisticR_G_mean(1,:)-logisticR_G_mean(Competitor_choice,:));
                            std5=stats5.sd;
                        case {7}%                 F_measure
                            [h1,p1,ci1,stats1] =ttest(KNN_F_measure(1,:),KNN_F_measure(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h2,p2,ci2,stats2] =ttest(DT_F_measure(1,:),DT_F_measure(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h3,p3,ci3,stats3] =ttest(SVM_F_measure(1,:),SVM_F_measure(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h4,p4,ci4,stats4] =ttest(NaiveB_F_measure(1,:),NaiveB_F_measure(Competitor_choice,:),'tail','right','alpha', alpha);
                            [h5,p5,ci5,stats5] =ttest(logisticR_F_measure(1,:),logisticR_F_measure(Competitor_choice,:),'tail','right','alpha', alpha);
                            
                            [h11,p11,ci11,stats11] =ttest(KNN_F_measure(1,:),KNN_F_measure(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h22,p22,ci22,stats22] =ttest(DT_F_measure(1,:),DT_F_measure(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h33,p33,ci33,stats33] =ttest(SVM_F_measure(1,:),SVM_F_measure(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h44,p44,ci44,stats44] =ttest(NaiveB_F_measure(1,:),NaiveB_F_measure(Competitor_choice,:),'tail','left','alpha', alpha);
                            [h55,p55,ci55,stats55] =ttest(logisticR_F_measure(1,:),logisticR_F_measure(Competitor_choice,:),'tail','left','alpha', alpha);
                            
                            mean1=mean(KNN_F_measure(1,:)-KNN_F_measure(Competitor_choice,:));
                            std1=stats1.sd;
                            mean2=mean(DT_F_measure(1,:)-DT_F_measure(Competitor_choice,:));
                            std2=stats2.sd;
                            mean3=mean(SVM_F_measure(1,:)-SVM_F_measure(Competitor_choice,:));
                            std3=stats3.sd;
                            mean4=mean(NaiveB_F_measure(1,:)-NaiveB_F_measure(Competitor_choice,:));
                            std4=stats4.sd;
                            mean5=mean(logisticR_F_measure(1,:)-logisticR_F_measure(Competitor_choice,:));
                            std5=stats5.sd;
                    end
                    
                    %依據檢定結果決定輸出  有顯著-附加*
                    fprintf(fileID,'%s\t%s\t',indexName{index_choice},method{Competitor_choice});
                    for class_choice=1:5
                        switch class_choice
                            case {1}
                                if h1==1
                                    sign='*';
                                else
                                    sign='';
                                end
                                if h11==1
                                    sign1='*';
                                else
                                    sign1='';
                                end
                                fprintf(fileID,'%s%f/%s%f/%f/%f\t',sign,p1,sign1,p11,mean1,std1);
                            case {2}
                                if h2==1
                                    sign='*';
                                else
                                    sign='';
                                end
                                if h22==1
                                    sign1='*';
                                else
                                    sign1='';
                                end
                                fprintf(fileID,'%s%f/%s%f/%f/%f\t',sign,p2,sign1,p22,mean2,std2);
                            case {3}
                                if h3==1
                                    sign='*';
                                else
                                    sign='';
                                end
                                if h33==1
                                    sign1='*';
                                else
                                    sign1='';
                                end
                                fprintf(fileID,'%s%f/%s%f/%f/%f\t',sign,p3,sign1,p33,mean3,std3);
                            case {4}
                                if h4==1
                                    sign='*';
                                else
                                    sign='';
                                end
                                if h44==1
                                    sign1='*';
                                else
                                    sign1='';
                                end
                                fprintf(fileID,'%s%f/%s%f/%f/%f\t',sign,p4,sign1,p44,mean4,std4);
                            case {5}
                                if h5==1
                                    sign='*';
                                else
                                    sign='';
                                end
                                if h55==1
                                    sign1='*';
                                else
                                    sign1='';
                                end
                                fprintf(fileID,'%s%f/%s%f/%f/%f\t\n',sign,p5,sign1,p55,mean5,std5);
                        end
                    end                    
                end
                fprintf(fileID,'\n');
            end
            fclose(fileID);
            
            
            
            fileID=fopen(strcat(trainFile{fileIndex},'-',outTextFileName,"檢定時的效能數字.txt"),'w');
            nbytes = fprintf(fileID,'%f\n',classRatio);
            nbytes = fprintf(fileID,'%d\t',delNo);
            nbytes = fprintf(fileID,'\n');
            
            fprintf(fileID,'\t\t%s\t%s\t%s\t%s\t%s\n','KNN','DT','SVM','NaiveB','LogitR');
            fprintf(fileID,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n','DataSet','Method','AUC','AUC','AUC','AUC','AUC');
            for i=1 : methodCount
                fprintf(fileID,'%s\t%s\t',trainFile{fileIndex},method{i});
                nbytes = fprintf(fileID,'%f\t',mean(KNN_AUC(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(DT_AUC(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(SVM_AUC(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(NaiveB_AUC(i,:)));
                nbytes = fprintf(fileID,'%f\n',mean(logisticR_AUC(i,:)));
            end
            
            fprintf(fileID,'\t\t%s\t%s\t%s\t%s\t%s\n','KNN','DT','SVM','NaiveB','LogitR');
            fprintf(fileID,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n','DataSet','Method','average','average','average','average','average');
            for i=1 : methodCount
                fprintf(fileID,'%s\t%s\t',trainFile{fileIndex},method{i});
                nbytes = fprintf(fileID,'%f\t',mean(KNN_Accurecy(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(DT_Accurecy(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(SVM_Accurecy(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(NaiveB_Accurecy(i,:)));
                nbytes = fprintf(fileID,'%f\n',mean(logisticR_Accurecy(i,:)));
            end
            
            fprintf(fileID,'\t\t%s\t%s\t%s\t%s\t%s\n','KNN','DT','SVM','NaiveB','LogitR');
            fprintf(fileID,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n','DataSet','Method','TP','TP','TP','TP','TP');
            for i=1 : methodCount
                fprintf(fileID,'%s\t%s\t',trainFile{fileIndex},method{i});
                nbytes = fprintf(fileID,'%f\t',mean(KNN_TP(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(DT_TP(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(SVM_TP(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(NaiveB_TP(i,:)));
                nbytes = fprintf(fileID,'%f\n',mean(logisticR_TP(i,:)));
            end
            
            fprintf(fileID,'\t\t%s\t%s\t%s\t%s\t%s\n','KNN','DT','SVM','NaiveB','LogitR');
            fprintf(fileID,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n','DataSet','Method','FP','FP','FP','FP','FP');
            for i=1 : methodCount
                fprintf(fileID,'%s\t%s\t',trainFile{fileIndex},method{i});
                nbytes = fprintf(fileID,'%f\t',mean(KNN_FP(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(DT_FP(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(SVM_FP(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(NaiveB_FP(i,:)));
                nbytes = fprintf(fileID,'%f\n',mean(logisticR_FP(i,:)));
            end
            
            fprintf(fileID,'\t\t%s\t%s\t%s\t%s\t%s\n','KNN','DT','SVM','NaiveB','LogitR');
            fprintf(fileID,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n','DataSet','Method','Precision','Precision','Precision','Precision','Precision');
            for i=1 : methodCount
                fprintf(fileID,'%s\t%s\t',trainFile{fileIndex},method{i});
                nbytes = fprintf(fileID,'%f\t',mean(KNN_Precision(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(DT_Precision(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(SVM_Precision(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(NaiveB_Precision(i,:)));
                nbytes = fprintf(fileID,'%f\n',mean(logisticR_Precision(i,:)));
            end
            
            fprintf(fileID,'\t\t%s\t%s\t%s\t%s\t%s\n','KNN','DT','SVM','NaiveB','LogitR');
            fprintf(fileID,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n','DataSet','Method','G_mean','G_mean','G_mean','G_mean','G_mean');
            for i=1 : methodCount
                fprintf(fileID,'%s\t%s\t',trainFile{fileIndex},method{i});
                nbytes = fprintf(fileID,'%f\t',mean(KNN_G_mean(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(DT_G_mean(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(SVM_G_mean(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(NaiveB_G_mean(i,:)));
                nbytes = fprintf(fileID,'%f\n',mean(logisticR_G_mean(i,:)));
            end
            
            fprintf(fileID,'\t\t%s\t%s\t%s\t%s\t%s\n','KNN','DT','SVM','NaiveB','LogitR');
            fprintf(fileID,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n','DataSet','Method','F_measure','F_measure','F_measure','F_measure','F_measure');
            for i=1 : methodCount
                fprintf(fileID,'%s\t%s\t',trainFile{fileIndex},method{i});
                nbytes = fprintf(fileID,'%f\t',mean(KNN_F_measure(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(DT_F_measure(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(SVM_F_measure(i,:)));
                nbytes = fprintf(fileID,'%f\t',mean(NaiveB_F_measure(i,:)));
                nbytes = fprintf(fileID,'%f\n',mean(logisticR_F_measure(i,:)));
            end
            fclose(fileID);
        end
    end
    
end

toc  %輸出程式執行時間
end

