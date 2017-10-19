function [ QSMOTEDateSet ] = QSMOTE3_1_1(  trainSet,majorClassNo,minorClassNo,K )
% 去除noise 後得到 sminfSet
% 以sminfSet 找到其最近K個多數鄰居聯集 得到 sbmajSet
% 以sbmajSet 找到其最近K個少數鄰居聯集 得到 siminSet
% 每一個少數類別(sminfSet) 找他最近的多數類別, 以此距離算出, 此少數類別附近的同伴有多少,並成為一個集合.
% 若集合與集合有相同元素,則合併此集合.
% 用SMOTE產生少數類別- 因為與群內點產生內插可能是錯誤的- 為了與QSMOTE3做比較
% 權重為 群內數的倒數
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
    majorNeighborNo=findRank(dist,data_sorted,(K-2));
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

G=majorClassInstanceCount-minorClassInstanceCount;
sumc=0;

for i=1 : size(sminfSet,2)
    S_dist=S_avg*10000;
    for j=1:size(sbmajSet,1)
        dist=NeighborDist(sminfSet(i),majorClassInstanceNo(sbmajSet(j)));
        if dist<S_dist
            S_dist=dist;              
        end       
    end
       
    member=[];
    for j=1:size(sminfSet,2)
        if(NeighborDist(sminfSet(i),minorClassInstanceNo(sminfSet(j)))==-1)
        else
            if(NeighborDist(sminfSet(i),minorClassInstanceNo(sminfSet(j)))<S_dist )
                member=union(member,minorClassInstanceNo(sminfSet(j)));
            end
        end
    end
    
    if i==1
        groupInfo={[minorClassInstanceNo(sminfSet(i)),member]};
    else
        groupInfo=[groupInfo,[minorClassInstanceNo(sminfSet(i)),member]];
    end
end

hasMerge=true;
while hasMerge
    groupI=-1;
    groupJ=-1;
    hasMerge=false;
    for i=1 : size(groupInfo,2)
        for j=1 : size(groupInfo,2)
            if (i~=j) && any(ismember(groupInfo{i},groupInfo{j}))
                hasMerge=true;
                groupI=i;
                groupJ=j;
                break;
            end
        end
        if hasMerge
            break;
        end
    end
    
    if hasMerge
        tempGroupInfo={union(groupInfo{groupI},groupInfo{groupJ})};
        for i=1 : size(groupInfo,2)
            if i==groupI || i==groupJ
            else
                tempGroupInfo=[tempGroupInfo,groupInfo{i}];
            end
        end
        groupInfo=tempGroupInfo;
    end
end

sumWeight=zeros(size(siminSet,1),0);
sumGroup=0;
for i=1 : size(siminSet,1)
    
    you=trainSet(minorClassInstanceNo(siminSet(i)),:);
    
    clusterNo=-1;
    for j=1 :size(groupInfo,2)
        if any(minorClassInstanceNo(siminSet(i))==groupInfo{j})
            clusterNo=j;
            break;
        end
    end
	sumGroup=sumGroup+(1/size(groupInfo{clusterNo},2));
	sumWeight(i)=(1/size(groupInfo{clusterNo},2));
end

for i=1 : size(siminSet,1)
    
    you=trainSet(minorClassInstanceNo(siminSet(i)),:);
    
	ratio=round((majorClassInstanceCount-size(sminfSet,2))*sumWeight(i)/sumGroup);
    
    temp=zeros(1,columnNum-1);
    for j=1:ratio
        randNo=randi([1 K],1,1);
        Neighbor=trainSet(kNeighborNo(siminSet(i),randNo),:);
        for k=1:columnNum-1
            temp(k)=(Neighbor(k)-you(k))*rand(1);
        end
        outDate=[you(1:columnNum-1)+temp,minorClassNo];
        outDataSet=[outDataSet;outDate];
    end   
end
QSMOTEDateSet=[trainSet;outDataSet];




end

