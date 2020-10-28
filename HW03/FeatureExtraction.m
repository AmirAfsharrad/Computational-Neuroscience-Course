function [feature,Index] = FeatureExtraction(Subject,Fs,string)
n = floor(0.8*Fs);
time = IndExtraction(Subject);

if (strcmp(string, 'test'))

Index = sort([time.Test_target, time.Test_nontarget]);
test_mat=zeros(n,11,length(Index));

for i = 1 : length(test_mat)
	test_mat(:,:,i) = Subject.test(:,Index(i):Index(i)+n-1)';
end
end

if (strcmp(string, 'train'))

Index = sort([time.Train_target, time.Train_nontarget]);
test_mat=zeros(n,11,length(Index));

for i = 1 : length(test_mat)
	test_mat(:,:,i) = Subject.train(:,Index(i):Index(i)+n-1)';
end
end

test_mat = permute(test_mat,[3,2,1]);

% feature = reshape(test_mat(:,2:9,:),[],8*n);
feature = myreshape(test_mat);
end

