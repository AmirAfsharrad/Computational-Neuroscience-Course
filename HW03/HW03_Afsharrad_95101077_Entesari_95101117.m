close all
clc
clear
%%
path=[pwd,'\DataSet'];
Data_folder=dir(path);
Fs=256;
%%
for i=3:length(Data_folder)
        Train_Prediction=struct2cell(load([path,'\',Data_folder(i).name]));
        num = str2double(Data_folder(i).name(2:strfind(Data_folder(i).name,'.')-1));
%         subject(num).number=str2double(Data_folder(i).name(2:strfind(Data_folder(i).name,'.')-1));
        subject(num).train=Train_Prediction{1}.train;
        subject(num).test=Train_Prediction{1}.test;
end
clear i a
%%  finding sc and rc
for i=1:10
    if any(subject(i).train(10,:)>12)
        subject(i).method='SC';
    else
        subject(i).method='RC';
    end
end
clear i
%% filtering all data
tic
bpf = BPF(1001,.5,30,256);
for i=1:10
    for j=2:9
        subject(i).train(j,:) = subject(i).train(j,:) - mean(subject(i).train(j,:));
        subject(i).train(j,:)=filter(bpf,1,subject(i).train(j,:));
        subject(i).test(j,:) = subject(i).test(j,:) - mean(subject(i).test(j,:));
        subject(i).test(j,:)=filter(bpf,1,subject(i).test(j,:));
    end
end
clear i j k s
toc
%% downsampling
for i=1:10
    Subject(i).method=subject(i).method;
    Subject(i).train(:,:)=subject(i).train(:,1:4:end);
    Subject(i).test(:,:)=subject(i).test(:,1:4:end);
end
% clear subject i j

%% IndExtracting
for i=1:10
    subject(i).time=IndExtraction(subject(i));
    Subject(i).time=IndExtraction(Subject(i));
end

%%  TrialExtraction
for i=1:10
[subject(i).train_target,subject(i).train_nontarget,...
    subject(i).test_target,subject(i).test_nontarget]=TrialExtraction(subject(i),Fs);
[Subject(i).train_target,Subject(i).train_nontarget,...
    Subject(i).test_target,Subject(i).test_nontarget]=TrialExtraction(Subject(i),Fs/4);
end

%% operating on subject number9, i,e. subject(10)
N = 4;
for n = 1 : 8
figure
t = linspace(0, 0.8, length(squeeze(mean(subject(N).train_target(:,1+n,:),1))));

MEAN_t = squeeze(mean(subject(N).train_target(:,1+n,:),1));
STD_t = squeeze(std(subject(N).train_target(:,1+n,:),1))/nthroot(size(subject(N).train_target(:,1+n,:),1),2);

MEAN_nt = squeeze(mean(subject(N).train_nontarget(:,1+n,:),1));
STD_nt = squeeze(std(subject(N).train_nontarget(:,1+n,:),1))/nthroot(size(subject(N).train_nontarget(:,1+n,:),1),2);

plot(t,MEAN_t,'r','LineWidth',2);
hold on
plot(t,MEAN_nt,'b','LineWidth',2);
plot(t,MEAN_t+STD_t,'r','LineWidth',0.5);
plot(t,MEAN_t-STD_t,'r','LineWidth',0.5);
plot(t,MEAN_nt+STD_nt,'b','LineWidth',0.5);
plot(t,MEAN_nt-STD_nt,'b','LineWidth',0.5);
title(['Subject #',num2str(N),' Electrode #',num2str(n)]);
legend('target','non-target');
xlabel('t(s)');
% ylim([-5,5]);
end

%% 4.1
% This part has already been done on previous sections and all data
% related to downsampled data is saved in struct 'Subject'

%% 4.2
np = floor(64*0.8);
for N = 1:10
    
% Train_target = reshape(Subject(N).train_target(:,2:9,:),[],np*8);
% Train_nontarget = reshape(Subject(N).train_nontarget(:,2:9,:),[],np*8);
Train_target = myreshape(Subject(N).train_target);
Train_nontarget = myreshape(Subject(N).train_nontarget);
Train_Feature = [Train_target; Train_nontarget];

% Test_target = reshape(Subject(N).test_target(:,2:9,:),[],np*8);
% Test_nontarget = reshape(Subject(N).test_nontarget(:,2:9,:),[],np*8);
Test_target = myreshape(Subject(N).test_target);
Test_nontarget = myreshape(Subject(N).test_nontarget);
Test_Feature = [Test_target; Test_nontarget];


labels = [ones(size(Train_target,1),1);zeros(size(Train_nontarget,1),1)];
clear Train_target Train_nontarget Test_target Test_nontarget

Subject(N).LDA_model = fitcdiscr(Train_Feature,labels);

Train_Prediction = predict(Subject(N).LDA_model,squeeze(Train_Feature));
Subject(N).Train_Percentage = 100*(1-sum(Train_Prediction~=labels)/length(labels))


Test_Prediction = predict(Subject(N).LDA_model,squeeze(Test_Feature));
Subject(N).Test_Percentage = 100*(sum(Test_Prediction==labels)/length(Test_Prediction))

CrossVal_obj = crossval(Subject(N).LDA_model,'KFold',5);
Subject(N).CrossVal = 100*(1-kfoldLoss(CrossVal_obj))

Test_Percentage(N) = 100*(sum(Test_Prediction(1:150)==1)/sum(Test_Prediction==1))

end
%% 4.3

for N = 1:10
    [Subject(N).trainWord] = WordRecognizer(Subject,N,Fs/4,'train');
    [Subject(N).testWord] = WordRecognizer(Subject,N,Fs/4,'test');
    
    [feature, I] = FeatureExtraction(Subject(N),Fs/4,'test');
    [Test_Prediction,score] = predict(Subject(N).LDA_model,feature);
%     Test_Prediction = zeros(size(score,1),1);
%     Test_Prediction(score(:,2)>0.8)=1;
    TestPredIndex = find(Test_Prediction == 1);
    
    [Subject(N).predictedWord]= WordRecognizer(Subject,N,Fs/4,'test',TestPredIndex);
    [Subject(N).predictedWord2]= WordRecognizer2(Subject,N,Fs/4,'test',TestPredIndex);
end

%% 4.4
for N = 1 : 10
    [~,I] = sort(abs(Subject(N).LDA_model.Coeffs(1,2).Linear));
    Subject(N).T = rem(I',51);
    Subject(N).Electrode = 1+floor(I'/51);
end

%% 4.5.a
%histogram of Test Correct Precentage
histogram([Subject.Test_Percentage],15);
title('Test Percentage Histogram');
xlabel('percentage')
%% 5
N = 1;
for i = 1 : size(Subject(N).train_target,1)
    for j = 2 : 9
        alpha_train_target(i,j-1,:) = filter(BPF(1001,8,15,64),1,Subject(N).train_target(i,j,:));
        beta_train_target(i,j-1,:) = filter(BPF(1001,15,31,64),1,Subject(N).train_target(i,j,:));
        alpha_test_target(i,j-1,:) = filter(BPF(1001,8,15,64),1,Subject(N).test_target(i,j,:));
        beta_test_target(i,j-1,:) = filter(BPF(1001,15,31,64),1,Subject(N).test_target(i,j,:));
    end
end


for i = 1 : size(Subject(N).train_nontarget,1)
    for j = 2 : 9
        alpha_train_nontarget(i,j-1,:) = filter(BPF(1001,8,15,64),1,Subject(N).train_nontarget(i,j,:));
        beta_train_nontarget(i,j-1,:) = filter(BPF(1001,15,31,64),1,Subject(N).train_nontarget(i,j,:));
        alpha_test_nontarget(i,j-1,:) = filter(BPF(1001,8,15,64),1,Subject(N).test_nontarget(i,j,:));
        beta_test_nontarget(i,j-1,:) = filter(BPF(1001,15,31,64),1,Subject(N).test_nontarget(i,j,:));
    end
end

L = floor (size(Subject(N).train_target,3)/5);
for i = 1 : size(Subject(N).train_target,1)
    for j = 1 : 8
        for k = 1 : L
            X1(i,j,k) = norm(squeeze(alpha_train_target(i,j,(k-1)*5+1:k*5)))^2;
        end
        for k = 1 : L
            X1(i,j,k+L) = norm(squeeze(beta_train_target(i,j,(k-1)*5+1:k*5)))^2;
        end
    end
end

L = floor (size(Subject(N).train_nontarget,3)/5);
for i = 1 : size(Subject(N).train_nontarget,1)
    for j = 1 : 8
        for k = 1 : L
            X2(i,j,k) = norm(squeeze(alpha_train_nontarget(i,j,(k-1)*5+1:k*5)))^2;
        end
        for k = 1 : L
            X2(i,j,k+L) = norm(squeeze(beta_train_nontarget(i,j,(k-1)*5+1:k*5)))^2;
        end
    end
end

X3 = reshape(X1,75,8*20);
X4 = reshape(X2,2625,8*20);

Feature = [X3;X4];
label = [ones(75,1);zeros(2625,1)];
clear X1 X2 X3 X4
a=[Feature, label];
