function [ average,TP_rate,FP_rate,Precision,AUC,G_mean,F_measure ] = returnPredition( test ,finalDecison,majorClassNo,minorClassNo)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    columnSize=size(test,2);
    count=0;
    TP_count=0;
    FP_count=0;
    for i=1:size(test,1)
        if finalDecison(i) == test(i,columnSize)
            count=count+1;
            
            if test(i,columnSize)==minorClassNo
               TP_count=TP_count+1;
            end
        else
            if test(i,columnSize)==majorClassNo
                FP_count=FP_count+1;
            end
        end  
        
    end
    average=count/size(test,1);
%     if isempty(find(test(:,columnSize)==minorClassNo, 1))
%         TP_rate=0.5;
%     else
        TP_rate=TP_count/length(find(test(:,columnSize)==minorClassNo));
%     end
    
%     if isempty(find(test(:,columnSize)==majorClassNo, 1))
%         FP_rate=0.5;
%     else
        FP_rate=FP_count/length(find(test(:,columnSize)==majorClassNo));    
%     end
    
    AUC=(1+TP_rate-FP_rate)/2;
    
    TN_rate=1-FP_rate;
    
    G_mean=sqrt(TP_rate*TN_rate);
    
    if (TP_count+FP_count)==0
        Precision=0;
    else
        Precision=TP_count/(TP_count+FP_count);
    end
    
    Recall=TP_rate;
    
    if (Precision+Recall)==0
        F_measure=0;
    else
        F_measure=(2*Precision*Recall)/(Precision+Recall);
    end
    [average,TP_rate,FP_rate,AUC,G_mean,F_measure];
end

