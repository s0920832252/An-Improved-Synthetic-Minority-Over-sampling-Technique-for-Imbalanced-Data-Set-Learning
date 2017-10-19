function [ Accurecy,TP,FP,Precision,AUC,G_mean,F_measure ] = logistic ( trainSet,testSet,majorClassNo,minorClassNo,featureNum )

CMdl =mnrfit(trainSet(:,1:featureNum-1), trainSet(:,featureNum));   
decisionA = mnrval(CMdl, testSet(:,1:featureNum-1));
decisionA(decisionA(:,1)>=0.5,1)=1;
decisionA(decisionA(:,1)<0.5,1)=2;
[tempAccurecy,tempTP,tempFP,tempPrecision,tempAUC,tempG_mean,tempF_measure ]=returnPredition(testSet,decisionA,majorClassNo,minorClassNo);

Accurecy=tempAccurecy;
TP=tempTP;
FP=tempFP;
Precision=tempPrecision;
AUC=tempAUC;
G_mean=tempG_mean;
F_measure=tempF_measure;



end

