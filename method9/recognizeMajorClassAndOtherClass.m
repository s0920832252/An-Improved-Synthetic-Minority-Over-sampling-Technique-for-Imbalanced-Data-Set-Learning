function [ Major,Minor ] = recognizeMajorClassAndOtherClass( Data )
% 辨認資料中的多數少數類別  並回傳其類別號碼
% 輸入: 資料   輸出: 多數類別以及少數類別的編號
%  目前僅實作 2class
    columnNumber= size(Data,2);
    rowNumber = size(Data,1);
    
    maxClass= max(Data(:,columnNumber));%知道類別號碼的 最大數字
    minClass= min(Data(:,columnNumber));%知道類別號碼的 最小數字
    majorCount=-1000000;
    minorCount=1000000;
    major=0;
    minor=0;
    tempCount=0;
    
    for i=minClass : maxClass
        tempCount=sum(Data(:,columnNumber)==i);
        if tempCount < minorCount
            minor=i;
            minorCount=tempCount;
        end
        
        if tempCount > majorCount
            major=i;
            majorCount=tempCount;
        end
    end
    Major=major;
    Minor=minor;

end

