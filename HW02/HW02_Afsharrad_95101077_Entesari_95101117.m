%% initial comments
% in order to have the code opetrate correctly, you have to add the
% functions BPF, AnnotExtract, and edfread to the search path or place
% them in the same folder as this code and you have to place all the data
% including edf and hypnogram files in a folder named data in the same
% directory as this code
% Good Luck:)
%% initialization
clc
clear
addpath('Data');
sub_num = 5;
% the lookup table for file adressing
lookup = {'ST7011J0-PSG.edf','ST7011JP-Hypnogram_annotations.txt';...
        'ST7022J0-PSG.edf','ST7022JM-Hypnogram_annotations.txt';...
        'ST7041J0-PSG.edf','ST7041JO-Hypnogram_annotations.txt';...
        'ST7052J0-PSG.edf','ST7052JA-Hypnogram_annotations.txt';...
        'ST7061J0-PSG.edf','ST7061JR-Hypnogram_annotations.txt'};
%% 3.2. loading the Data 
[t, s, X] = FeatureExtraction(lookup{sub_num,1},lookup{sub_num,2});
classificationSet = [X s];

%% 3.2 performing PCA
[coeff,score,latent] = pca(X);
%% 3.2 plotting cumulative energy (eigenvalues) 
plot(100*cumsum(latent)/sum(latent));
xlabel('N');
ylabel('Percentage of Var.');
title(['subject ',num2str(sub_num)]);
%% 3.3 Extracting different states as pints in 3D space
SW = X(s == 0,:)*coeff(:,1:3);
S1 = X(s == 1,:)*coeff(:,1:3);
S2 = X(s == 2,:)*coeff(:,1:3);
S3 = X(s == 3,:)*coeff(:,1:3);
S4 = X(s == 4,:)*coeff(:,1:3);
SR = X(s == 6,:)*coeff(:,1:3);

%% 3.3. PC1 - PC2 - PC3 3d-plot
figure
subplot(221)
plot3(SW(:,1),SW(:,2),SW(:,3),'.');
hold on
plot3(S1(:,1),S1(:,2),S1(:,3),'.');
plot3(S2(:,1),S2(:,2),S2(:,3),'.');
plot3(S3(:,1),S3(:,2),S3(:,3),'.');
plot3(S4(:,1),S4(:,2),S4(:,3),'.');
plot3(SR(:,1),SR(:,2),SR(:,3),'.');
title(['subject',num2str(sub_num)]);
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
legend('W','1','2','3','R');
x1 = -2000000; x2 = 6000000;
y1 = -500000; y2 = 1000000;
z1 = -500000; z2 = 500000;
xlim([x1,x2]); ylim([y1,y2]); zlim([z1,z2]);
%% 3.3 PC1 and PC2
subplot(222)
plot(SW(:,1),SW(:,2),'.');
hold on
plot(S1(:,1),S1(:,2),'.');
plot(S2(:,1),S2(:,2),'.');
plot(S3(:,1),S3(:,2),'.');
plot(S4(:,1),S4(:,2),'.');
plot(SR(:,1),SR(:,2),'.');
title(['subject',num2str(sub_num)]);
xlabel('PC1'); ylabel('PC2');
legend('W','1','2','3','4','R');

xlim([x1,x2]); ylim([y1,y2]);
%% 3.3 PC1 and PC3
subplot(223)
plot(SW(:,1),SW(:,3),'.');
hold on
plot(S1(:,1),S1(:,3),'.');
plot(S2(:,1),S2(:,3),'.');
plot(S3(:,1),S3(:,3),'.');
plot(S4(:,1),S4(:,3),'.');
plot(SR(:,1),SR(:,3),'.');
title(['subject',num2str(sub_num)]);
xlabel('PC1'); ylabel('PC3');
legend('W','1','2','3','4','R');
a = 100;
b = 10000;
xlim([x1,x2]); ylim([z1,z2]);
%% 3.3 PC2 and PC3

subplot(224)
plot(SW(:,2),SW(:,3),'.');
hold on
plot(S1(:,2),S1(:,3),'.');
plot(S2(:,2),S2(:,3),'.');
plot(S3(:,2),S3(:,3),'.');
plot(S4(:,2),S4(:,3),'.');
plot(SR(:,2),SR(:,3),'.');
title(['subject',num2str(sub_num)]);
xlabel('PC2'); ylabel('PC3');
legend('W','1','2','3','4','R');
a = 100;
b = 10000;
xlim([y1,y2]); ylim([z1,z2]);
%% 3.4. a new plot for understanding the significant vector
figure
subplot(18,10,1:9)
plot(SW(:,1),zeros(1,length(SW(:,1))),'x')
title(['Subject ',num2str(sub_num)]);
subplot(18,10,10)
text(0,0.5,'SW,PC1','interpreter','latex'); axis off
subplot(18,10,11:19)
plot(SW(:,2),zeros(1,length(SW(:,2))),'x')
subplot(18,10,20)
text(0,0.5,'SW,PC2','interpreter','latex'); axis off
subplot(18,10,21:29)
plot(SW(:,3),zeros(1,length(SW(:,3))),'x')
subplot(18,10,30)
text(0,0.5,'SW,PC3','interpreter','latex'); axis off
subplot(18,10,31:39)
plot(SR(:,1),zeros(1,length(SR(:,1))),'x')
subplot(18,10,40)
text(0,0.5,'SR,PC1','interpreter','latex'); axis off
subplot(18,10,41:49)
plot(SR(:,2),zeros(1,length(SR(:,2))),'x')
subplot(18,10,50)
text(0,0.5,'SR,PC2','interpreter','latex'); axis off
subplot(18,10,51:59)
plot(SR(:,3),zeros(1,length(SR(:,3))),'x')
subplot(18,10,60)
text(0,0.5,'SR,PC3','interpreter','latex'); axis off
subplot(18,10,61:69)
plot(S1(:,1),zeros(1,length(S1(:,1))),'x')
subplot(18,10,70)
text(0,0.5,'S1,PC1','interpreter','latex'); axis off
subplot(18,10,71:79)
plot(S1(:,2),zeros(1,length(S1(:,2))),'x')
subplot(18,10,80)
text(0,0.5,'S1,PC2','interpreter','latex'); axis off
subplot(18,10,81:89)
plot(S1(:,3),zeros(1,length(S1(:,3))),'x')
subplot(18,10,90)
text(0,0.5,'S1,PC3','interpreter','latex'); axis off
subplot(18,10,91:99)
plot(S2(:,1),zeros(1,length(S2(:,1))),'x')
subplot(18,10,100)
text(0,0.5,'S2,PC1','interpreter','latex'); axis off
subplot(18,10,101:109)
plot(S2(:,2),zeros(1,length(S2(:,2))),'x')
subplot(18,10,110)
text(0,0.5,'S2,PC2','interpreter','latex'); axis off
subplot(18,10,111:119)
plot(S2(:,3),zeros(1,length(S2(:,3))),'x')
subplot(18,10,120)
text(0,0.5,'S2,PC3','interpreter','latex'); axis off
subplot(18,10,121:129)
plot(S3(:,1),zeros(1,length(S3(:,1))),'x')
subplot(18,10,130)
text(0,0.5,'S3,PC1','interpreter','latex'); axis off
subplot(18,10,131:139)
plot(S3(:,2),zeros(1,length(S3(:,2))),'x')
subplot(18,10,140)
text(0,0.5,'S3,PC2','interpreter','latex'); axis off
subplot(18,10,141:149)
plot(S3(:,3),zeros(1,length(S3(:,3))),'x')
subplot(18,10,150)
text(0,0.5,'S3,PC3','interpreter','latex'); axis off
subplot(18,10,151:159)
plot(S4(:,1),zeros(1,length(S4(:,1))),'x')
subplot(18,10,160)
text(0,0.5,'S4,PC1','interpreter','latex'); axis off
subplot(18,10,161:169)
plot(S4(:,2),zeros(1,length(S4(:,2))),'x')
subplot(18,10,170)
text(0,0.5,'S4,PC2','interpreter','latex'); axis off
subplot(18,10,171:179)
plot(S4(:,3),zeros(1,length(S4(:,3))),'x')
subplot(18,10,180)
text(0,0.5,'S4,PC3','interpreter','latex'); axis off
%% 3.4. histograms for sleep depth recognition based on PC1, PC2, and PC3
figure
n = 120;
x1 = -100; x2 = 500;
y1=0; y2 = 5000;
z1 = 0; z2 = 2000;
subplot(3,1,1)
histogram(S1(:,1),linspace(x1,x2,n),'normalization','pdf');
title('PC1')
hold on
histogram(S2(:,1),linspace(x1,x2,n),'normalization','pdf');
histogram(S3(:,1),linspace(x1,x2,n),'normalization','pdf');
histogram(S4(:,1),linspace(x1,x2,n),'normalization','pdf');
legend('1','2','3','4')
xlim([x1,x2]);

subplot(3,1,2)
histogram(S1(:,2),linspace(y1,y2,n),'normalization','pdf');
title('PC2')
hold on
histogram(S2(:,2),linspace(y1,y2,n),'normalization','pdf');
histogram(S3(:,2),linspace(y1,y2,n),'normalization','pdf');
histogram(S4(:,2),linspace(y1,y2,n),'normalization','pdf');
legend('1','2','3','4')
xlim([y1,y2]);

subplot(3,1,3)
histogram(S1(:,3),linspace(z1,z2,n),'normalization','pdf');
hold on
title('PC3')
histogram(S2(:,3),linspace(z1,z2,n),'normalization','pdf');
histogram(S3(:,3),linspace(z1,z2,n),'normalization','pdf');
histogram(S4(:,3),linspace(z1,z2,n),'normalization','pdf');
legend('1','2','3','4')
xlim([z1,z2]);

%% 4.1. removing states 0 and 6 from matrix X and creating the linear model
Index = (0<s)&(s<5);
new_s = s(Index);
new_X = X(Index,:);
% new_X = zscore(X(Index,:));
tbl = table(new_X(:,1),new_X(:,2),new_X(:,3),new_X(:,4),new_X(:,5),...
    new_X(:,6),new_X(:,7),new_X(:,8),new_X(:,9),new_X(:,10),s(Index),...
    'VariableNames',{'Oz_delta','Oz_theta','Oz_alpha','Oz_beta',...
    'Fpz_delta','Fpz_theta','Fpz_alpha','Fpz_beta','EOG','EMG','state'});
LinMod = fitlm(tbl);
%% 4.4. histograms for the 4 sleep depths (fitted)
bin = 0.1;
histogram(LinMod.Fitted(new_s == 1),0:bin:5,'Normalization','pdf');
title(['Subject ',num2str(sub_num)]);
grid on
hold on
histogram(LinMod.Fitted(new_s == 2),0:bin:5,'Normalization','pdf');
histogram(LinMod.Fitted(new_s == 3),0:bin:5,'Normalization','pdf');
histogram(LinMod.Fitted(new_s == 4),0:bin:5,'Normalization','pdf');
legend('1','2','3','4');
%% 4.5. histograms for the 2 new states (wake and REM) (predicted)
figure
wake_Index = s==0;
REM_Index = s==6;
histogram(LinMod.predict(X(wake_Index,:)),0:bin:5,'Normalization','pdf');
title(['Subject ',num2str(sub_num)]);
grid on
hold on
histogram(LinMod.predict(X(REM_Index,:)),0:bin:5,'Normalization','pdf');
legend('WAKE','REM');
%% 5.2 clustering
cluster = kmeans(X,4)-1;
cluster0 = cluster(s==0);
cluster1 = cluster(s==1);
cluster2 = cluster(s==2);
cluster3 = cluster(s==3);
cluster4 = cluster(s==4);
cluster6 = cluster(s==6);
subplot(6,1,1)
stem(cluster0)
ylim([0,5])
subplot(6,1,2)
stem(cluster1)
ylim([0,5])
subplot(6,1,3)
stem(cluster2)
ylim([0,5])
subplot(6,1,4)
stem(cluster3)
ylim([0,5])
subplot(6,1,5)
stem(cluster4)
ylim([0,5])
subplot(6,1,6)
stem(cluster6)
ylim([0,5])
%% 5.3 adding memory to the problem
% the code in part %% 3.2. loading the Data %% must be substituted with the
% following code and the rest shall be run exactly the same wat
[t, s, X] = FeatureExtraction(lookup{sub_num,1},lookup{sub_num,2});
k = 4;
for i = k+1 : length(s)
    memX(i-k,:) = mean(X(i-k:i,:),1);
end
X = memX;
s = s(k+1:end);
classificationSet = [X s];
