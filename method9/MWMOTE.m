function [ MWMOTEDateSet ] = MWMOTE( trainSet,minorClassNo,N,K1,K2,K3 )

outDataSet=[];  %用來裝輸出的資料集
columnNum=size(trainSet,2);  %所有feature數(包括class feature)
rowNum=size(trainSet,1); %所有資料筆數
minorClassInstanceNo=find(trainSet(:,columnNum)==minorClassNo); %少數類別資料在全部資料內的編號
majorClassInstanceNo=setdiff(1:rowNum,minorClassInstanceNo);  %多數類別資料在全部資料內的編號
minorClassInstanceCount= size(minorClassInstanceNo,1);     %少數類別資料數
majorClassInstanceCount=rowNum-minorClassInstanceCount;     %多數類別資料數
sminfSet=[]; %不包含鄰居全為多數類別的少數類別集合
sbmajSet=[]; %邊界多數類別集合
siminSet=[]; %邊界少數類別集合
CMAX=2;  %函數參數
Cfth=5;  %函數參數
Cp=0.05;    %函數參數
kNeighborNo_K1=zeros(minorClassInstanceCount,K1);   %紀錄每一個少數類別的最近k1個鄰居編號
temp=zeros(1,columnNum-1);  %做內插之暫存值(一row資料缺label)

%預先找出每一個少數類別實例的最近k1個鄰居編號-用來刪除noise
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
    NeighborNo=findRank(dist,data_sorted,K1+1);
    kNeighborNo_K1(i,:)=NeighborNo(2:K1+1); %只記錄最近k1個鄰居 (排除自己)
    
    %判斷是否全為多數類別鄰居, 紀錄不是的, 以消除noise
    majorCount=0;
    for j=1 : K1
        if trainSet(kNeighborNo_K1(i,j),columnNum)~=minorClassNo
            majorCount=majorCount+1;
        end
    end
    if majorCount~=K1
        sminfSet=[sminfSet,i];
    end
end


% kNeighborNo_K2=zeros(size(sminfSet,2),K2);   %紀錄每一個少數類別的最近k2個鄰居編號
% 找到Sbmaj,邊界多數類別
for i=1 : size(sminfSet,2)
    dist=zeros(majorClassInstanceCount,1);   %預先初始化-執行速度會快一點
    for j=1 : majorClassInstanceCount
        dist(j)=norm(trainSet(majorClassInstanceNo(j),1:columnNum-1)-trainSet(minorClassInstanceNo(sminfSet(i)),1:columnNum-1),2);
    end
    
    data_sorted=sort(dist);    %排序此少數類別實例的多數類別鄰居,由最近排到最遠
    majorNeighborNo=findRank(dist,data_sorted,K2);
    
    %     kNeighborNo_K2(i,:)=majorNeighborNo(1:K2); %記錄sminfSet的最近k2個多數類別鄰居
    sbmajSet=union (sbmajSet,majorNeighborNo(1:K2));
end

% kNeighborNo_K3=zeros(size(sbmajSet,1),K3);
% 找到Simin,邊界少數類別
% for i=1 : size(sbmajSet,1)
%     dist=zeros(minorClassInstanceCount,1);   %預先初始化-執行速度會快一點
%     for j=1 : minorClassInstanceCount
%         dist(j)=norm(trainSet(majorClassInstanceNo(sbmajSet(i)),1:columnNum-1)-trainSet(minorClassInstanceNo(j),1:columnNum-1),2);
%     end
%     
%     data_sorted=sort(dist);
%     minNeighborNo=findRank(dist,data_sorted,K3);
%     %     kNeighborNo_K3(i,:)=minNeighborNo(1:K3);
%     siminSet=union (siminSet,minNeighborNo(1:K3));
% end

% 找到Simin,邊界少數類別
for i=1 : size(sbmajSet,1)
    dist=zeros(size(sminfSet,2),1);   %預先初始化-執行速度會快一點
    for j=1 : size(sminfSet,2)
        dist(j)=norm(trainSet(majorClassInstanceNo(sbmajSet(i)),1:columnNum-1)-trainSet(minorClassInstanceNo(sminfSet(j)),1:columnNum-1),2);
    end
    data_sorted=sort(dist);  %排序此少數類別實例的所有鄰居,由最近排到最遠
    [~, minNeighborNo]=ismember(dist,data_sorted);
    siminSet=union (siminSet,sminfSet(minNeighborNo(1:K3)));
end



%計算所有的Cf值
Cf=zeros(size(siminSet,1),size(sbmajSet,1));
for i=1 : size(siminSet,1)
    for j=1 : size(sbmajSet,1)
        dn=norm(trainSet(majorClassInstanceNo(sbmajSet(j)),1:columnNum-1)-trainSet(minorClassInstanceNo(siminSet(i)),1:columnNum-1),2)/columnNum;
        x=1/dn;
        if x<=Cfth
            Cf(i,j)=x/Cfth*CMAX;
        else
            Cf(i,j)=1*CMAX;
        end
    end
end

%計算所有的Iw值
Iw=zeros(size(Cf,1),size(Cf,2));
for i=1 : size(siminSet,1)
    for j=1 : size(sbmajSet,1)
        Iw(i,j)=Cf(i,j)*Cf(i,j)/sum(Cf(i,:));
    end
end

%算出Sw值
Sw=zeros(size(siminSet,1),1);
for i=1 : size(siminSet,1)
    Sw(i)=sum(Iw(i,:));
end

%Clustering 區域
% A=[1,2,3]
% B=[4,5]
% C={A,B}
% C{2}(1) ->  4
minorInstanceArray=zeros(minorClassInstanceCount,minorClassInstanceCount);
for i=1 : minorClassInstanceCount
    for j=1 : minorClassInstanceCount
        if i==j
            minorInstanceArray(i,j)=-1;
        else
            minorInstanceArray(i,j)=norm(trainSet(minorClassInstanceNo(i),1:columnNum-1)-trainSet(minorClassInstanceNo(j),1:columnNum-1),2);
        end
    end
end

for i=1 : minorClassInstanceCount
    if i==1
        groupInfo={[i]};
    else
        groupInfo=[groupInfo,[i]];
    end    
end
closeDis=0;
davg=0;
for i=1 : size(sminfSet,2)
    for j=1: size(sminfSet,2)
        if i~=j
            davg=davg+minorInstanceArray(sminfSet(i),sminfSet(j));
        end
    end
end
davg=davg/(2*size(sminfSet,2));
Th=davg*Cp;
while size(groupInfo,2)>1
    %     disArray=zeros(size(groupInfo,2),size(groupInfo,2));
    
    minAvgDistGroup=100000000;
    pairI=-1;
    pairJ=-1;
    for i=1 : size(groupInfo,2)
        for j=1 : size(groupInfo,2)
            if i==j
                %                 disArray(i,j)=-1;
            else
                sumDistGroup=0;
                for k=1 : size(groupInfo{i},2)
                    for l=1 : size(groupInfo{j},2)
                        if groupInfo{i}(k)~=groupInfo{j}(l)
                            sumDistGroup=sumDistGroup+minorInstanceArray(groupInfo{i}(k),groupInfo{j}(l));
                        end
                    end
                end
                avgDistGroup=sumDistGroup/(size(groupInfo{i},2)*size(groupInfo{j},2));
                %                 disArray(i,j)=avgDistGroup;
                if avgDistGroup<minAvgDistGroup
                    pairI=i;
                    pairJ=j;
                    minAvgDistGroup=avgDistGroup;
                end
            end
        end
    end
    
    closeDis=minAvgDistGroup;
    
    if closeDis > Th
        break;
    end
    
    tempGroupInfo={union(groupInfo{pairI},groupInfo{pairJ})};
    for i=1 : size(groupInfo,2)
        if i==pairI || i==pairJ
        else
            tempGroupInfo=[tempGroupInfo,groupInfo{i}];
        end
    end
    groupInfo=tempGroupInfo;
end

%Clustering

instanceCount=0;
while instanceCount<N
    SampleNo=randsample(1:size(siminSet,1),1,true,Sw);
    you=trainSet(minorClassInstanceNo(siminSet(SampleNo)),:);
    
    clusterNo=-1;
    for i=1 :size(groupInfo,2)
        if any(minorClassInstanceNo(siminSet(SampleNo))==minorClassInstanceNo(groupInfo{i}))
            clusterNo=i;
            break;
        end
    end
    
    randNo=randi([1 size(groupInfo{clusterNo},2) ],1,1);
    Neighbor=trainSet(groupInfo{clusterNo}(randNo),:);
    
    %執行內插
    for k=1:columnNum-1
        temp(k)=(Neighbor(k)-you(k))*rand(1);
    end
    outDate=[you(1:columnNum-1)+temp,minorClassNo]; %新產生的資料實例標註為少數類別
    outDataSet=[outDataSet;outDate];    %將新產生的資料實例放入輸出陣列中    
    instanceCount=instanceCount+1;
end

MWMOTEDateSet=[trainSet;outDataSet];




end

