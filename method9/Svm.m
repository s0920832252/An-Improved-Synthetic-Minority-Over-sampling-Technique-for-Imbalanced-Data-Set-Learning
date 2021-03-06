function [ Accurecy,TP,FP,Precision,AUC,G_mean,F_measure ] = Svm( trainSet,testSet,majorClassNo,minorClassNo,featureNum )

% CMdl = fitcsvm(trainSet(:,1:featureNum-1),trainSet(:,featureNum),'Standardize',true ,'Solver','SMO','CacheSize','maximal','KernelFunction','gaussian');
CMdl = fitcsvm(trainSet(:,1:featureNum-1),trainSet(:,featureNum),'Standardize',true );
decisionA = predict(CMdl,testSet(:,1:featureNum-1));
[tempAccurecy,tempTP,tempFP,tempPrecision,tempAUC,tempG_mean,tempF_measure ]=returnPredition(testSet,decisionA,majorClassNo,minorClassNo);

Accurecy=tempAccurecy;
TP=tempTP;
FP=tempFP;
Precision=tempPrecision;
AUC=tempAUC;
G_mean=tempG_mean;
F_measure=tempF_measure;

end

