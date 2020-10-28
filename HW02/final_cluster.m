function y = final_cluster(correlation, threshold, Index, n)
mat = zeros(n,length(correlation),length(correlation));
for i = 1 : n
    a = cluster(correlation, threshold, Index);
    for j = 1 : length(a)
        if(length(a{j})>1)
            for k = 1 : length(a{j})-1
                mat(i,a{j}(k),a{j}(k+1)) = 1;
                mat(i,a{j}(k+1),a{j}(k)) = 1;
            end
        end
    end
end
y1 = eye(length(correlation))+reshape(mean(mat,1),length(Index),length(Index));
y = cluster(y1, eps, Index);