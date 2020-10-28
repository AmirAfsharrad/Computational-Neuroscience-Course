function error_percentage = Final_CrossVal(TrainData, labels, K, n)
    Error = zeros(1,n);
    for i = 1 : n    
        Error(i) = CrossVal(TrainData, labels, K);
    end
    error_percentage = mean(Error);
end
