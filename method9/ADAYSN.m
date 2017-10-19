function [ ADAYSNDataSet ] = ADAYSN( trainSet,minorClassNo,K )

outDataSet=[];  %用來裝輸出的資料集
columnNum=size(trainSet,2);  %所有feature數(包括class feature)
rowNum=size(trainSet,1); %所有資料筆數
minorClassInstanceNo=find(trainSet(:,columnNum)==minorClassNo); %少數類別資料在全部資料內的編號
minorClassInstanceCount= size(minorClassInstanceNo,1);     %少數類別資料數
majorClassInstanceCount=rowNum-minorClassInstanceCount;      %多數類別資料數
kNeighborNo=zeros(minorClassInstanceCount,K);   %紀錄每一個少數類別的最近k個鄰居編號
majorNeighborCountSet=zeros(1,minorClassInstanceCount);
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


% 算出每一個少數類別實例的多數類別鄰居數
for i=1 : minorClassInstanceCount
    majorNeighborCount=0;
    for j=1 : K
        if trainSet(kNeighborNo(i,j),columnNum)~=minorClassNo
            majorNeighborCount=majorNeighborCount+1;
        end
    end
    majorNeighborCountSet(i)=majorNeighborCount;
end

G=majorClassInstanceCount-minorClassInstanceCount;
for i=1 : minorClassInstanceCount
    you=trainSet(minorClassInstanceNo(i),:);
    
    for j=1:(majorNeighborCountSet(i)*G/sum(majorNeighborCountSet))
        minorKNeighborNo=[]; %鄰居編號-專存少數類別
        for k=1 : K
            if trainSet(kNeighborNo(i,k),columnNum)==minorClassNo
                minorKNeighborNo=[minorKNeighborNo,k];
            end
        end
        if size(minorKNeighborNo,2)==0  %沒有少數類別鄰居      
        else
            rN=randi([1 size(minorKNeighborNo,2)],1,1);        
            Neighbor=trainSet( kNeighborNo(i,minorKNeighborNo(rN)),:); %取出鄰居的資料
            
            for k=1:columnNum-1
                temp(k)=(Neighbor(k)-you(k))*rand(1);
            end
            outDate=[you(1:columnNum-1)+temp,minorClassNo]; %新產生的資料實例標註為少數類別
            outDataSet=[outDataSet;outDate];    %將新產生的資料實例放入輸出陣列中            
        end         
    end        
end

ADAYSNDataSet=[trainSet;outDataSet];
end


% randsample(1:2,100,true,[p1 p2 p3 p4 p5 p6])
