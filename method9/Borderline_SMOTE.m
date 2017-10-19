function [ Borderline_SMOTEDataSet ] = Borderline_SMOTE( trainSet,minorClassNo,K )

outDataSet=[];  %用來裝輸出的資料集
columnNum=size(trainSet,2);  %所有feature數(包括class feature)
rowNum=size(trainSet,1); %所有資料筆數
minorClassInstanceNo=find(trainSet(:,columnNum)==minorClassNo); %少數類別資料在全部資料內的編號
minorClassInstanceCount= size(minorClassInstanceNo,1);     %少數類別資料數
kNeighborNo=zeros(minorClassInstanceCount,K);   %紀錄每一個少數類別的最近k個鄰居編號
DangerSet=[];   %危險集,用來存放製造內插的種子以及其多數鄰居個數
temp=zeros(1,columnNum-1);  %做內插之暫存值(一row資料缺label)


%預先找出每一個少數類別實例的最近k個鄰居編號
for i=1 : minorClassInstanceCount
    dist=zeros(rowNum,1);   %預先初始化-執行速度會快一點
    %計算出此少數類別實例到所有資料實例的距離(使用2-norm)
    for j=1:rowNum
        if j==minorClassInstanceNo(i)
            dist(j)=-1;
        else
            dist(j)=norm(trainSet(j,1:columnNum-1)-trainSet(minorClassInstanceNo(i),1:columnNum-1),2);
        end
    end
    data_sorted=sort(dist);  %排序此少數類別實例的所有鄰居,由最近排到最遠
    NeighborNo=findRank(dist,data_sorted,K+1);
    kNeighborNo(i,:)=NeighborNo(2:K+1); %只記錄最近k個鄰居 (排除自己)
end

%將邊界成員加入危險集.
for i=1 : minorClassInstanceCount
    majorNeighborCount=0;
    for j=1 : K
        if trainSet(kNeighborNo(i,j),columnNum)~=minorClassNo
            majorNeighborCount=majorNeighborCount+1;
        end
    end
    if majorNeighborCount>=(K/2) && majorNeighborCount<K
        DangerSet=[DangerSet;i];
    end
end

% 使用危險集合內的成員做內插
for i=1 : size(DangerSet,1)
    minorKNeighborNo=[];
    %判斷此危險集合成員的哪一個鄰居是少數類別
    for j=1 : K
        if trainSet(kNeighborNo(DangerSet(i),j),columnNum)==minorClassNo
            minorKNeighborNo=[minorKNeighborNo,j];
        end
    end
    s=size(minorKNeighborNo,2);
    for q=1 : s
        Neighbor=trainSet(kNeighborNo(DangerSet(i),minorKNeighborNo(q)),:);
        you=trainSet(minorClassInstanceNo(DangerSet(i)),:);
        for k=1:columnNum-1
            temp(k)=(Neighbor(k)-you(k))*rand(1);
        end
        outDate=[you(1:columnNum-1)+temp,minorClassNo]; %新產生的資料實例標註為少數類別
        outDataSet=[outDataSet;outDate];    %將新產生的資料實例放入輸出陣列中
    end
end

Borderline_SMOTEDataSet=[trainSet;outDataSet]; %回傳 (輸出陣列+原始資料)
end

