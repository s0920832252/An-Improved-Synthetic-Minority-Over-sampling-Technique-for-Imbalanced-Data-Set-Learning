function [ MSNOTEDataSet ] = MSNOTE( trainSet,minorClassNo,N,K )

outDataSet=[];  %用來裝輸出的資料集
columnNum=size(trainSet,2);  %所有feature數(包括class feature)
rowNum=size(trainSet,1); %所有資料筆數
minorClassInstanceNo=find(trainSet(:,columnNum)==minorClassNo); %少數類別資料在全部資料內的編號
minorClassInstanceCount= size(minorClassInstanceNo,1);     %少數類別資料數
minorClassInstanceType=zeros(minorClassInstanceCount,1);    %少數類別資料的類型-noise/borider/safe
kNeighborNo=zeros(minorClassInstanceCount,K);   %紀錄每一個少數類別的最近k個鄰居編號
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

%預先找出每一個少數類別的類型
for i=1 : minorClassInstanceCount
    majorKNeighborCount=0;
    for j=1 : K
        if trainSet(kNeighborNo(i,j),columnNum)~=minorClassNo
            majorKNeighborCount=majorKNeighborCount+1;
        end
    end
    
    if majorKNeighborCount==K
        minorClassInstanceType(i)=1; %noise
    elseif majorKNeighborCount==0
        minorClassInstanceType(i)=2; %safe
    else
        minorClassInstanceType(i)=3; %boundary
    end
end

for i=1 : minorClassInstanceCount
    if minorClassInstanceType(i)~=1
        for j=1 : N
            if minorClassInstanceType(i)==2
                rN=randi([1 K],1,1);
                Neighbor=trainSet(kNeighborNo(rN),:);   %取出鄰居的資料
            elseif minorClassInstanceType(i)==3
                Neighbor=trainSet(kNeighborNo(1),:);   %取出最近鄰居的資料
            end
            
            you=trainSet(minorClassInstanceNo(i),:);    %此少數類別實例的資料
            %執行內插
            for k=1:columnNum-1
                temp(k)=(Neighbor(k)-you(k))*rand(1);
            end
            outDate=[you(1:columnNum-1)+temp,minorClassNo]; %新產生的資料實例標註為少數類別
            outDataSet=[outDataSet;outDate];    %將新產生的資料實例放入輸出陣列中
        end
    end
end

MSNOTEDataSet=[trainSet;outDataSet]; %回傳 (輸出陣列+原始資料)

end

