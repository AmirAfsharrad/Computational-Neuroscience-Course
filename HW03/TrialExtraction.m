function [Train_tar_mat,Train_non_mat,Test_tar_mat,Test_non_mat] = TrialExtraction(subject,Fs)
n = floor(0.8*Fs);
time = IndExtraction(subject);
Train_tar_mat=zeros(n,11,length(time.Train_target));
Train_non_mat=zeros(n,11,length(time.Train_nontarget));

Test_tar_mat=zeros(n,11,length(time.Test_target));
Test_non_mat=zeros(n,11,length(time.Test_nontarget));

for i = 1 : length(time.Train_target)
	Train_tar_mat(:,:,i) = subject.train(:,time.Train_target(i):time.Train_target(i)+n-1)';
end

for i = 1 : length(time.Train_nontarget)
	Train_non_mat(:,:,i) = subject.train(:,time.Train_nontarget(i):time.Train_nontarget(i)+n-1)';
end

for i = 1 : length(time.Test_target)
	Test_tar_mat(:,:,i) = subject.test(:,time.Test_target(i):time.Test_target(i)+n-1)';
end

for i = 1 : length(time.Test_nontarget)
	Test_non_mat(:,:,i) = subject.test(:,time.Test_nontarget(i):time.Test_nontarget(i)+n-1)';
end
Train_non_mat=permute(Train_non_mat,[3,2,1]);
Train_tar_mat=permute(Train_tar_mat,[3,2,1]);
Test_non_mat=permute(Test_non_mat,[3,2,1]);
Test_tar_mat=permute(Test_tar_mat,[3,2,1]);
end

