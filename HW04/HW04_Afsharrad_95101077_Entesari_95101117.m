%% 3.1 calculating t-values and z-values for each voxel
clc
addpath('E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\spm12');
addpath('E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\niitools');
for n = 1 : 7
    addpath(['E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\run0' ...
        num2str(n)]);
    % all data is saved in struct 'data'
    % saving t-values
    [~,data(n).t.rock] = convertnii2mat(['spmT_0001.nii'],'untouch');
    [~,data(n).t.symphonic] = convertnii2mat(['spmT_0002.nii'],'untouch');
    [~,data(n).t.metal] = convertnii2mat(['spmT_0003.nii'],'untouch');
    [~,data(n).t.country] = convertnii2mat(['spmT_0004.nii'],'untouch');
    [~,data(n).t.ambient] = convertnii2mat(['spmT_0005.nii'],'untouch');
    % saving z-values using spm_t2z
    data(n).z.rock = spm_t2z(data(n).t.rock,140);
    data(n).z.symphonic = spm_t2z(data(n).t.symphonic,140);
    data(n).z.metal = spm_t2z(data(n).t.metal,140);
    data(n).z.country = spm_t2z(data(n).t.country,140);
    data(n).z.ambient = spm_t2z(data(n).t.ambient,140);
    close all
    rmpath(['E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\run0' ...
        num2str(n)]);
end
%% 3.1 plotting some histograms for t-values and z-values to achieve a better knowledge of the dataset
% histograms of t-values and z-values including t = 0 and z = 0
subplot(2,2,1)
histogram(data(1).t.rock)
title('histogram of t-values for run#1')
subplot(2,2,2)
histogram(data(1).z.rock)
title('histogram of z-values for run#1')
subplot(2,2,3)
histogram(data(7).t.rock)
title('histogram of t-values for run#7')
subplot(2,2,4)
histogram(data(7).z.rock)
title('histogram of z-values for run#7')

% histograms of t-values and z-values excluding t = 0 and z = 0
figure
subplot(2,2,1)
histogram(data(1).t.rock(data(1).t.rock~=0))
title('histogram of t-values for run#1')
subplot(2,2,2)
histogram(data(1).z.rock(data(1).z.rock~=0))
title('histogram of z-values for run#1')
subplot(2,2,3)
histogram(data(7).t.rock(data(7).t.rock~=0))
title('histogram of t-values for run#7')
subplot(2,2,4)
histogram(data(7).z.rock(data(7).z.rock~=0))
title('histogram of z-values for run#7')
%% 3.2 averaging over runs
avg_data.rock = data(1).z.rock;
avg_data.symphonic = data(1).z.symphonic;
avg_data.metal = data(1).z.metal;
avg_data.country = data(1).z.country;
avg_data.ambient = data(1).z.ambient;
for i = 2 : 7
    avg_data.rock = avg_data.rock + data(i).z.rock;
    avg_data.symphonic = avg_data.symphonic + data(i).z.symphonic;
    avg_data.metal = avg_data.metal + data(i).z.metal;
    avg_data.country = avg_data.country + data(i).z.country;
    avg_data.ambient = avg_data.ambient + data(i).z.ambient;
end
avg_data.rock = avg_data.rock/7;
avg_data.symphonic = avg_data.symphonic/7;
avg_data.metal = avg_data.metal/7;
avg_data.country = avg_data.country/7;
avg_data.ambient = avg_data.ambient/7;
%% 3.2 finding a good threshold for plotting the active parts of the brain
k = 3;
threshold = k*std([reshape(avg_data.rock,1,[])...
    reshape(avg_data.symphonic,1,[]) ...
    reshape(avg_data.metal,1,[])...
    reshape(avg_data.country,1,[])...
    reshape(avg_data.ambient,1,[])]);

avg_data.rock_th = avg_data.rock;
avg_data.rock_th(abs(avg_data.rock - mean(mean(mean(avg_data.rock)))) < threshold ) = 0;

avg_data.symphonic_th = avg_data.symphonic;
avg_data.symphonic_th(abs(avg_data.symphonic - mean(mean(mean(avg_data.symphonic)))) < threshold ) = 0;

avg_data.metal_th = avg_data.metal;
avg_data.metal_th(abs(avg_data.metal - mean(mean(mean(avg_data.metal)))) < threshold ) = 0;

avg_data.country_th = avg_data.country;
avg_data.country_th(abs(avg_data.country - mean(mean(mean(avg_data.country)))) < threshold ) = 0;

avg_data.ambient_th = avg_data.ambient;
avg_data.ambient_th(abs(avg_data.ambient - mean(mean(mean(avg_data.ambient)))) < threshold ) = 0;
%% 3.2 plotting the active parts of brain while playing different genres
brainPlot(abs(avg_data.rock_th))
brainPlot(abs(avg_data.symphonic_th))
brainPlot(abs(avg_data.metal_th))
brainPlot(abs(avg_data.country_th))
brainPlot(abs(avg_data.ambient_th))
%% 3.3 loading z-values corresponding to the differential contrasts
for n = 1 : 7
    addpath(['E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\run0' ...
        num2str(n)]);
    % all data is saved in struct 'data'
    % saving t-values
    for i = 6 : 15
        if (i<10)
            [~,DifferentialData_t{n,i}] = convertnii2mat(['spmT_000' num2str(i) '.nii'],'untouch');
        else
            [~,DifferentialData_t{n,i}] = convertnii2mat(['spmT_00' num2str(i) '.nii'],'untouch');
        end
        close all
    end
    
    % saving z-values using spm_t2z
    for i = 1 : 15
            DifferentialData_z{n,i} = spm_t2z(DifferentialData_t{n,i},140);
    end
    
    rmpath(['E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\run0' ...
        num2str(n)]);
end
%% 3.3 averaging over runs and plotting the difference of active parts of brain between different genres
meanZvalue(1).name = 'R-S';
meanZvalue(2).name = 'R-M';
meanZvalue(3).name = 'R-C';
meanZvalue(4).name = 'R-A';
meanZvalue(5).name = 'S-M';
meanZvalue(6).name = 'S-C';
meanZvalue(7).name = 'S-A';
meanZvalue(8).name = 'M-C';
meanZvalue(9).name = 'M-A';
meanZvalue(10).name = 'C-A';
for i = 1 : 10
    meanZvalue(i).z = (DifferentialData_z{1,i+5}+DifferentialData_z{2,i+5}+DifferentialData_z{3,i+5}+...
        DifferentialData_z{4,i+5}+DifferentialData_z{5,i+5}+DifferentialData_z{6,i+5}+...
        DifferentialData_z{7,i+5})/7;
    
    brainPlot(abs(meanZvalue(i).z))
end

%% 4.1 creating the Feature Matrix using z-values
lookup = ['R' 'S' 'M' 'C' 'A'];
allFeature = zeros(7*25, 160*160*36 + 2);

for n = 1 : 7
    addpath(['E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\classification\run0' ...
        num2str(n)]);
    for i = 1 : 5
        for j = 1 : 5
            if ((i-1)*5 + j < 10)
                [~,temp] = convertnii2mat(['spmT_000' num2str((i-1)*5 + j) '.nii'],'untouch');
            else
                [~,temp] = convertnii2mat(['spmT_00' num2str((i-1)*5 + j) '.nii'],'untouch');
            end
            z_temp = spm_t2z(temp,120);
            allFeature((n-1)*25 + (i-1)*5 + j, 1:end-2) = reshape(z_temp, 1, []);
            allFeature((n-1)*25 + (i-1)*5 + j, end-1) = lookup(i);
            allFeature((n-1)*25 + (i-1)*5 + j, end) = n;
        end
    end
    rmpath(['E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\classification\run0' ...
        num2str(n)]);
    close all
end
%% 4.1 plotting some histograms for z-values to achieve a better knowledge of the dataset
subplot(2,2,1)
histogram(allFeature(1,1:end-2))
title('histogram of z-values, genre: rock, run#1')

subplot(2,2,2)
histogram(allFeature(31,1:end-2))
title('histogram of z-values, genre: symphonic, run#2')

subplot(2,2,3)
histogram(allFeature(61,1:end-2))
title('histogram of z-values, genre: metal, run#3')

subplot(2,2,4)
histogram(allFeature(86,1:end-2))
title('histogram of z-values, genre: country, run#4')

%% 4.1 removing voxels having all z-values equal to zero
zeroVoxels = sum(allFeature == 0,1);
nonzeroVoxels = ~(zeroVoxels==175);
Feature = allFeature(:,nonzeroVoxels);
FinalTest = FinalTest(:,nonzeroVoxels);
clear allFeature nonzeroVoxels zeroVoxels

%% 4.1 plotting some histograms for z-values to achieve a better knowledge of the dataset after removing zero voxels
subplot(2,2,1)
histogram(Feature(1,1:end-2))
title('histogram of z-values, genre: rock, run#1')

subplot(2,2,2)
histogram(Feature(31,1:end-2))
title('histogram of z-values, genre: symphonic, run#2')

subplot(2,2,3)
histogram(Feature(61,1:end-2))
title('histogram of z-values, genre: metal, run#3')

subplot(2,2,4)
histogram(Feature(86,1:end-2))
title('histogram of z-values, genre: country, run#4')
%% 4.2 calculating the p-values corresponding to each voxel, using ANOVA hypothesis testing
pval = zeros(1,size(Feature,2)-2);
for i = 1 : size(Feature,2)-2
    pval(i) = anova1(Feature(:,i),Feature(:,end-1),'off');
end
%%
histogram(pval)
title('histogram of p-values corresponding to all voxels')
%% 4.3 creating an LDA model for p = 0.01 and calculating the correct percentage
threshold = 0.01;
reduced_feature = Feature(:,[pval<threshold logical([1 1])]);
X = reduced_feature(:,1:end-2);
Y = reduced_feature(:,end-1);
LDAmodel = fitcdiscr (X, Y);
result = predict(LDAmodel, X);
AccuracyOnTrainData = sum(result == Y)*100/length(Y)
clear correct_percentage LDAmodel reduced_feature result threshold X Y
%% 4.4 & 4.5 calculating p-values for cross-validation, each time removing one of the 7 runs
for k = 1 : 7
    Feature_Cval = Feature(Feature(:,end) ~= k,:);
    for i = 1 : size(Feature_Cval,2)-2
        pval_Cval(k,i) = anova1(Feature_Cval(:,i),Feature_Cval(:,end-1),'off');
    end
end
%% 4.4 & 4.5 performing 7-fold cross-validation for thresholds from p = 5*10^-5 to p = 10^-2
threshold = 0.00005 : 0.00001 : 0.01;
for i = 1 : 996
    for k = 1 : 7
        Feature_Cval = Feature(Feature(:,end) ~= k,:);
        reduced_feature_Cval = Feature_Cval(:,[pval_Cval(k,:)<threshold(i) logical([1 1])]);
        TrainData = reduced_feature_Cval(:,1:end-2);
        TrainLabel = reduced_feature_Cval(:,end-1);
        LDAmodel = fitcdiscr (TrainData, TrainLabel);
        TestData = Feature(Feature(:,end) == k, pval_Cval(k,:)<threshold(i));
        TestLabel = Feature(Feature(:,end) == k, end-1);
        result = predict(LDAmodel, TestData);
        correct_percentage(i,k) = sum(result == TestLabel)*100/length(result);
    end
    i
end
%% 4.4 & 4.5 plot of cross-validation result against thresholds
mean_correct_percentage = mean(correct_percentage,2);
std_correct_percentage = std(correct_percentage,0,2);
plot(threshold,mean_correct_percentage);
hold on
plot(threshold,mean_correct_percentage+std_correct_percentage);
plot(threshold,mean_correct_percentage-std_correct_percentage);
legend('cross-validation mean','cross-validation mean + std','cross-validation mean - std')
xlabel ('p-value')
ylabel('correct percentage')
title('cross-validation correct percentage against p-value threshold')
%% 4.4 & 4.5 finding the best threshold, the corresponding number of voxels, and confusion matrix
[~,i] = max(mean_correct_percentage);
optimum_threshold = threshold(i)
MaximumAccuracy = mean_correct_percentage(i);
StandardDeviation = std_correct_percentage(i);
for k = 1 : 7
    Feature_Cval = Feature(Feature(:,end) ~= k,:);
    reduced_feature_Cval = Feature_Cval(:,[pval_Cval(k,:)<threshold(i) logical([1 1])]);
    number_of_voxels(k) = size(reduced_feature_Cval,2);
    TrainData = reduced_feature_Cval(:,1:end-2);
    TrainLabel = reduced_feature_Cval(:,end-1);
    LDAmodel = fitcdiscr (TrainData, TrainLabel);
    TestData = Feature(Feature(:,end) == k, pval_Cval(k,:)<threshold(i));
    TestLabel = Feature(Feature(:,end) == k, end-1);
    result = predict(LDAmodel, TestData);
    for j = 1 : 5
        temp = result == lookup(j);
        confmat(k,j,1) = 100*sum(temp.*(TestLabel==lookup(1)))/sum((TestLabel==lookup(1)));
        confmat(k,j,2) = 100*sum(temp.*(TestLabel==lookup(2)))/sum((TestLabel==lookup(2)));
        confmat(k,j,3) = 100*sum(temp.*(TestLabel==lookup(3)))/sum((TestLabel==lookup(3)));
        confmat(k,j,4) = 100*sum(temp.*(TestLabel==lookup(4)))/sum((TestLabel==lookup(4)));
        confmat(k,j,5) = 100*sum(temp.*(TestLabel==lookup(5)))/sum((TestLabel==lookup(5)));
    end
end
optimum_threshold
number_of_voxels
MaximumAccuracy
StandardDeviation 
confusion = squeeze(mean(confmat,1));
%% 4.7 creating crossval structure to save test and train for each of the 7 runs
for k = 1 : 7
    Feature_Cval = Feature(Feature(:,end) ~= k,:);
    crossval(k).feature = Feature_Cval(:,[pval_Cval(k,:)<optimum_threshold logical([1 1])]);
    crossval(k).TestFeature = Feature(Feature(:,end) == k, pval_Cval(k,:)<optimum_threshold);
    crossval(k).TestLabel = Feature(Feature(:,end) == k, end-1);
end
%% 4.7 Logistic Regression for the whole Dataset
LogisticRegCoeffs = mnrfit(Feature(:,pval<optimum_threshold),categorical(Feature(:,end-1)));
result = mnrval(LogisticRegCoeffs, Feature(:,pval<optimum_threshold));
[~,result] = max(result,[],2);
lookup = 'ACMRS';
LogisticRegressionAccuracy = 100*sum(lookup(result)' == Feature(:,end-1))/175

%% 4.7 Logistic Regression Corss-Validation for the optimum threshold found in previous sections (p = 0.0038)
% creating the estimators for Multinomial Logistic Regression using mnrfit
for k = 1 : 7
    feature = crossval(k).feature;
    tic
    crossval(k).LogisticRegCoeffs = mnrfit(feature(:,1:end-2),categorical(feature(:,end-1)));
    toc
end
% performing the cross-validation using mnrval
lookup = 'ACMRS';
for k = 1 : 7
    result = mnrval(crossval(k).LogisticRegCoeffs, crossval(k).TestFeature);
    [~,result] = max(result,[],2);
    LogisticRegressionCrossValAccuracy(k) = 100*sum(lookup(result)' == crossval(k).TestLabel)/25;
end
LogisticRegressionCrossValAccuracy
%% 4.7 SVM for the whole Dataset
SVMmodel = fitcecoc(Feature(:,pval<optimum_threshold),Feature(:,end-1));
result = predict(SVMmodel, Feature(:,pval<optimum_threshold));
SVMAccuracy = 100*sum(result == Feature(:,end-1))/175

%% 4.7 SVM Cross-Validation for thresholds from p = 5*10^-5 to p = 2*10^-2
threshold = 0.00005 : 0.00001 : 0.02;
for i = 1301 : 1996
    i
for k = 1 : 7
    Feature_Cval = Feature(Feature(:,end) ~= k,:);
    SVMmodel = fitcecoc(Feature_Cval(:,pval_Cval(k,:)<threshold(i)),Feature_Cval(:,end-1));
    result = predict(SVMmodel,Feature(Feature(:,end) == k, pval_Cval(k,:)<threshold(i)));
    SVMCrossValAccuracy(i,k) = 100*sum(result == Feature(Feature(:,end) == k, end-1))/25;
end
end

plot(threshold,mean(SVMCrossValAccuracy,2))
hold on
plot(threshold,mean(SVMCrossValAccuracy,2)+std(SVMCrossValAccuracy,0,2))
plot(threshold,mean(SVMCrossValAccuracy,2)-std(SVMCrossValAccuracy,0,2))
legend('cross-validation mean','cross-validation mean + std','cross-validation mean - std')
xlabel ('p-value')
ylabel('correct percentage')
title('cross-validation correct percentage against p-value threshold')
%% 4.8 confusion matrix for the optimum threshold
% this part has already been done on sections 4.4 and 4.5
%% 4.9 extracting the feature matrix for run#8
FinalTest = zeros(25,921602);
n = 8;
    addpath(['E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\classification\run08'])
    for i = 1 : 5
        for j = 1 : 5
            if ((i-1)*5 + j < 10)
                [~,temp] = convertnii2mat(['spmT_000' num2str((i-1)*5 + j) '.nii'],'untouch');
            else
                [~,temp] = convertnii2mat(['spmT_00' num2str((i-1)*5 + j) '.nii'],'untouch');
            end
            z_temp = spm_t2z(temp,120);
            FinalTest((i-1)*5 + j, 1:end-2) = reshape(z_temp, 1, []);
            FinalTest((i-1)*5 + j, end-1) = lookup(i);
            FinalTest((i-1)*5 + j, end) = n;
        end
    end
    rmpath(['E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW04\CN_HW04\classification\run08']);
    close all




%% 4.8
threshold = optimum_threshold;
reduced_feature = Feature(:,[pval<threshold logical([1 1])]);
reducedFinalTest = FinalTest(:,pval<threshold);
X = reduced_feature(:,1:end-2);
Y = reduced_feature(:,end-1);
LDAmodel = fitcdiscr (X, Y);
temp = predict(LDAmodel, reducedFinalTest);
for i = 1 : 25
    switch (temp(i))
        case 'R'
            Predicted_Label{i} = 'rock';
        case 'S'
            Predicted_Label{i} = 'symphonic';
        case 'M'
            Predicted_Label{i} = 'metal';
        case 'C'
            Predicted_Label{i} = 'country';
        case 'A'
            Predicted_Label{i} = 'ambient';
    end
end


%% 5 Feature Reduction based on J-value and cross-valdiation
NumberOfFeatures = 10 : 10 : 4000;
tic
for k = 1 : 7
        Jcrossval(k).J (1,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'R','S');
        Jcrossval(k).J (2,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'R','M');
        Jcrossval(k).J (3,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'R','C');
        Jcrossval(k).J (4,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'R','A');
        Jcrossval(k).J (5,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'S','M');
        Jcrossval(k).J (6,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'S','C');
        Jcrossval(k).J (7,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'S','A');
        Jcrossval(k).J (8,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'M','C');
        Jcrossval(k).J (9,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'M','A');
        Jcrossval(k).J (10,:) = Jvalue(Feature(Feature(:,end)~=k,1:end),'C','A');
end
%% 
for i = 201 : 400
    i
    for k = 1 : 7
        n = NumberOfFeatures(i)/10;
        [~,Jcrossval(k).Index] = sort(Jcrossval(k).J,2,'descend');
        
        Jcrossval(k).FeatureIndex = [Jcrossval(k).Index(1,1:n) Jcrossval(k).Index(2,1:n)...
            Jcrossval(k).Index(3,1:n) Jcrossval(k).Index(4,1:n) Jcrossval(k).Index(5,1:n)...
            Jcrossval(k).Index(6,1:n) Jcrossval(k).Index(7,1:n) Jcrossval(k).Index(8,1:n) ...
            Jcrossval(k).Index(9,1:n) Jcrossval(k).Index(10,1:n)];
        Jcrossval(k).ReducedFeature = Feature(Feature(:,end)~=k,Jcrossval(k).FeatureIndex );
        Jcrossval(k).TrainLabel = Feature(Feature(:,end)~=k,end-1);
        
        Jcrossval(k).TestFeature = Feature(Feature(:,end)==k,Jcrossval(k).FeatureIndex );
        Jcrossval(k).TestLabel = Feature(Feature(:,end)==k,end-1);
        
        Jcrossval(k).LDAmodel = fitcdiscr(Jcrossval(k).ReducedFeature, Jcrossval(k).TrainLabel);
        Jcrossval(k).SVMmodel = fitcecoc(Jcrossval(k).ReducedFeature, Jcrossval(k).TrainLabel);
        
        Jcrossval(k).predictedLDA = predict(Jcrossval(k).LDAmodel,Jcrossval(k).TestFeature);
        Jcrossval(k).predictedSVM = predict(Jcrossval(k).SVMmodel,Jcrossval(k).TestFeature);
        
        Jcrossval(k).TruePercentageLDA = 100*sum(Jcrossval(k).predictedLDA == Jcrossval(k).TestLabel)/25;
        Jcrossval(k).TruePercentageSVM = 100*sum(Jcrossval(k).predictedSVM == Jcrossval(k).TestLabel)/25;
        
        
    end
    JTruePercentage_SVM(i) = mean([Jcrossval.TruePercentageSVM]);
        JTruePercentageSTD_SVM(i) = std([Jcrossval.TruePercentageSVM]);
        
        JTruePercentage_LDA(i) = mean([Jcrossval.TruePercentageLDA]);
        JTruePercentageSTD_LDA(i) = std([Jcrossval.TruePercentageLDA]);
end
toc
%% 5 Cross-Validation Results based on number of features using J-value-based feature selection
plot(NumberOfFeatures,JTruePercentage_LDA)
hold on
plot(NumberOfFeatures,JTruePercentage_LDA+JTruePercentageSTD_LDA)
plot(NumberOfFeatures,JTruePercentage_LDA-JTruePercentageSTD_LDA)

legend('cross-validation mean','cross-validation mean + std','cross-validation mean - std')
xlabel ('number of Features')
ylabel('correct percentage')
title('cross-validation correct percentage against number of Features (LDA-based)')


figure
plot(NumberOfFeatures,JTruePercentage_SVM)
hold on
plot(NumberOfFeatures,JTruePercentage_SVM+JTruePercentageSTD_SVM)
plot(NumberOfFeatures,JTruePercentage_SVM-JTruePercentageSTD_SVM)

legend('cross-validation mean','cross-validation mean + std','cross-validation mean - std')
xlabel ('number of Features')
ylabel('correct percentage')
title('cross-validation correct percentage against number of Features (SVM-based)')
