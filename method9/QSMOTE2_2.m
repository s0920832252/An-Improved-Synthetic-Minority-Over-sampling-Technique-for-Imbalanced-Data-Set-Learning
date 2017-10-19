function [ QSMOTEDateSet ] = QSMOTE2_2(  trainSet,majorClassNo,minorClassNo,K )
% 去除noise 後得到 sminfSet
% 以sminfSet 找到其最近K個多數鄰居聯集 得到 sbmajSet
% 以sbmajSet 找到其最近K個少數鄰居聯集 得到 siminSet
% 對siminSet成員計算 W 以及 距離  ,並以此為SEED SET
% W= 多數類別同伴數+1(1是自己)/少數類別同伴數+1
% 距離= 多數類別同半數1/ 多+少 同半數+1+1


outDataSet=[];  %用來裝輸出的資料集
columnNum=size(trainSet,2);  %所有feature數(包括class feature)
rowNum=size(trainSet,1); %所有資料筆數
minorClassInstanceNo=find(trainSet(:,columnNum)==minorClassNo); %少數類別資料在全部資料內的編號
majorClassInstanceNo=setdiff(1:rowNum,minorClassInstanceNo);  %多數類別資料在全部資料內的編號
minorClassInstanceCount= size(minorClassInstanceNo,1);     %少數類別資料數
majorClassInstanceCount=rowNum-minorClassInstanceCount;     %多數類別資料數
sminfSet=[]; %不包含鄰居全為多數類別的少數類別集合
sbmajSet=[]; %邊界多數類別集合
siminSet=[];
majorNeighborCountSet=zeros(1,minorClassInstanceCount);
kNeighborNo=zeros(minorClassInstanceCount,K);   %紀錄每一個少數類別的最近k個鄰居編號
NeighborDist=zeros(minorClassInstanceCount,rowNum);
S=0;

Dist=zeros(rowNum,rowNum);
for i=1:rowNum
    for j=1:rowNum
        if i==j
            Dist(j)=-1;
        else
            Dist(i,j)=norm(trainSet(j,1:columnNum-1)-trainSet(i,1:columnNum-1),2);
        end
    end
end


%預先找出每一個少數類別實例的最近k個鄰居編號
for i=1 : minorClassInstanceCount
    dist=zeros(rowNum,1);   %預先初始化-執行速度會快一點
    %計算出此少數類別實例到所有資料實例的距離(使用2-norm)
    for j=1:rowNum
        if j==minorClassInstanceNo(i)
            dist(j)=-1;
        else
            dist(j)=norm(trainSet(j,1:columnNum-1)-trainSet(minorClassInstanceNo(i),1:columnNum-1),2);
            if trainSet(j,columnNum)==majorClassNo
                S=S+dist(j);
            end
        end
        NeighborDist(i,j)=dist(j);
    end
    data_sorted=sort(dist);  %排序此少數類別實例的所有鄰居,由最近排到最遠
    NeighborNo=findRank(dist,data_sorted,K+1);
    kNeighborNo(i,:)=NeighborNo(2:K+1); %只記錄最近k個鄰居 (排除自己)
    
    %判斷是否全為多數類別鄰居, 紀錄不是的, 以消除noise
    majorCount=0;
    for j=1 : K
        if trainSet(kNeighborNo(i,j),columnNum)~=minorClassNo
            majorCount=majorCount+1;
        end
    end
    if majorCount~=K
        sminfSet=union(sminfSet,i);
    end
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

S_avg=S/(majorClassInstanceCount*minorClassInstanceCount);

% 找到Sbmaj,邊界多數類別
for i=1 : size(sminfSet,2)
    dist=zeros(majorClassInstanceCount,1);   %預先初始化-執行速度會快一點
    for j=1 : majorClassInstanceCount
        dist(j)=NeighborDist(sminfSet(i),majorClassInstanceNo(j));
    end
    data_sorted=sort(dist);  %排序此少數類別實例的所有鄰居,由最近排到最遠
    majorNeighborNo=findRank(dist,data_sorted,K);
    sbmajSet=union (sbmajSet,majorNeighborNo);
end

% 找到Simin,邊界少數類別
for i=1 : size(sbmajSet,1)
    dist=zeros(size(sminfSet,2),1);   %預先初始化-執行速度會快一點
    for j=1 : size(sminfSet,2)
        dist(j)=NeighborDist(sminfSet(j),majorClassInstanceNo(sbmajSet(i)));
    end
    data_sorted=sort(dist);  %排序此少數類別實例的所有鄰居,由最近排到最遠
    [~, minNeighborNo]=ismember(dist,data_sorted);
    siminSet=union (siminSet,sminfSet(minNeighborNo(1:(K-2))));
end



W=zeros(size(sminfSet,2),1);
DW=zeros(size(sminfSet,2),1);
for i=1 :size(siminSet,1)
    S_dist=S_avg*10000;
    majNo=0;
    for j=1:size(sbmajSet,1)
        dist=NeighborDist(siminSet(i),majorClassInstanceNo(sbmajSet(j)));
        if dist<S_dist
            S_dist=dist;
            majNo=majorClassInstanceNo(sbmajSet(j));
        end
    end
    
    majorCount=0;
    for j=1 : majorClassInstanceCount
        if Dist(majNo,majorClassInstanceNo(j))==-1
        else
            dist=Dist(majNo,majorClassInstanceNo(j));
            if dist<S_dist
                majorCount=majorCount+1;
            end
        end
    end
    
    minorCount=0;
    for j=1:minorClassInstanceCount
        if(NeighborDist(siminSet(i),minorClassInstanceNo(j))==-1)
        else
            if(NeighborDist(siminSet(i),minorClassInstanceNo(j))<S_dist )
                minorCount=minorCount+1;
            end
        end
    end
    W(i)=(majorCount+1)/(minorCount+1);
%     W(i)=majorCount/(minorCount+1);
    DW(i)=(majorCount+1)/(majorCount+minorCount+1+1); 
%     DW(i)=(minorCount+1)/(majorCount+minorCount+1); 
end

sumW=sum(W);

for i=1 : size(siminSet,1)
    you=trainSet(minorClassInstanceNo(siminSet(i)),:);
    
    S_dist=S_avg*10000;
    majNo=0;
    for j=1:size(sbmajSet,1)
        dist=NeighborDist(siminSet(i),majorClassInstanceNo(sbmajSet(j)));
        if dist<S_dist
            S_dist=dist;
            majNo=majorClassInstanceNo(sbmajSet(j));
        end
    end
    
    
    Neighbor=trainSet(majNo,:);    
    
    ratio=round((majorClassInstanceCount-size(sminfSet,2))*W(i)/sumW);
    
    temp=zeros(1,columnNum-1);
    for j=1:ratio        
        for k=1:columnNum-1
            temp(k)=(Neighbor(k)-you(k))*rand(1)*DW(i);
        end
        outDate=[you(1:columnNum-1)+temp,minorClassNo];
        outDataSet=[outDataSet;outDate];
    end
    
end

QSMOTEDateSet=[trainSet;outDataSet];




end

