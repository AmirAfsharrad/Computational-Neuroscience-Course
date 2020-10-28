function y = cluster(correlation, threshold, Index)
    correlation = abs(correlation);
	check = 1;
    min = 1;
    y = cell(0);
    Index = Index(randperm(length(Index)));
    for i = 1 : length(Index)
        for j = i : length(Index)
            if(correlation(Index(i),Index(j))<threshold)
                check = 0;
                if(correlation(Index(i),Index(j)) < min)
                    min = correlation(Index(i),Index(j));
                    I1 = Index(i);
                    I2 = Index(j);
                end
            end
        end
    end
    if(check || (length(Index)==1))
        y = [y, Index];
    else
        Index1 = I1;
        Index2 = I2;
        for i =  length(Index): -1 : 1
            if((Index(i) ~= I1) && (Index(i) ~= I2))
                L1 = max(correlation(Index(i),Index1));
                L2 = max(correlation(Index(i),Index2));
                if (L1>L2)
                    Index1 = [Index1, Index(i)];
                else
                    Index2 = [Index2, Index(i)];
                end
            end
        end
%         Index1
%         Index2
        y = [y, cluster(correlation, threshold, Index1), cluster(correlation, threshold, Index2)];
% y=[y, Index1, Index2];
%             cluster(correlation, threshold, Index1);
%             cluster(correlation, threshold, Index2);
    end
end