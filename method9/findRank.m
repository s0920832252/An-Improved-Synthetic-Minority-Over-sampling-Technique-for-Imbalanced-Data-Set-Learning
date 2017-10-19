function [ rank ] = findRank( dist,data_sorted,K )
    tempRank=zeros(K,1);
    for i=1:K
        for j=1:length(dist)
            if dist(j)==data_sorted(i)
                tempRank(i)=j;
            end
        end
    end
    rank=tempRank;
end

