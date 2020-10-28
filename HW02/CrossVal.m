function error_percentage = CrossVal(TrainData, labels, K)
    number_of_samples = size(TrainData,1);
    Index = randperm(number_of_samples);
    default_length = floor(number_of_samples/K);
    L = default_length*ones(1,K);
    L(K) = number_of_samples - (K-1)*default_length;
    Error = zeros(1,K);
    for i = 1 : K
        Train = [TrainData(Index(1:sum(L(1:i-1))),:);TrainData(Index(sum(L(1:i))+1:end),:)];
        label = [labels(Index(1:sum(L(1:i-1)))) labels(Index(sum(L(1:i))+1:end))];
        Test = TrainData(Index(sum(L(1:i-1))+1:sum(L(1:i))),:);
        True_Answers = labels(Index(sum(L(1:i-1))+1:sum(L(1:i))));
        result = MultiSVM(Train,label,Test);
        Error(i) = sum(True_Answers ~= result')/length(result);
    end
    error_percentage = 100*mean(Error);
end