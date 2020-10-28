%% 2.2
clear
CurrentFolder = pwd;
path = [CurrentFolder,'\Data\Spike_and_Log_Files'];
file_list = dir(path);
    for i = 3:63            %creating struct "neuron" to save important data
        neurons(i-2).name = file_list(i).name;
        [neurons(i-2).Data, neurons(i-2).freq, neurons(i-2).rate] = ...
            Func_ReadData(file_list(i).name);  
        neurons(i-2).included = 1;
        if neurons(i-2).rate < 2
            neurons(i-2).included = 0;
            display (neurons(i-2).name);
        end
    end
    histogram([neurons.rate],0:2:26);
    title('Spike Rate Histogram','interpreter','latex');
% clear file_list i path
%% 2.5
addpath('E:\SharifUniversityOfTechnology\96_2\Computational_Nueroscience\HW01\MatlabFunctions\tview');
cd Data/Spike_and_Log_Files/000412.a01
tview('000412.a01atune.log');
cd ../000524.c01
tview('000524.c01atune.log');
cd ../../..
%% Neuron used for parts 3.1,3.2,3.3,3.4,3.5 is 000601.c05
%% 3.1 & 3.2
n = 16;             % neurons number in struct "neurons"
load([CurrentFolder,'\Data\Stimulus_Files\msq1D.mat'])
Spike_trig_stimuli = Func_StimuliExtraction ([neurons(n).Data.events],msq1D,neurons(n).freq);
% Spike triggered ensemble
Spike_trig_stimuli = permute(Spike_trig_stimuli,[3,1,2]);
% the receptive field
rec_field = reshape(mean(Spike_trig_stimuli),16,16);
[~, p1] = ttest (Spike_trig_stimuli);
p1 = 1-p1;
p1 = reshape(p1,16,16);

figure
subplot(2,2,1)
imshow(rec_field,[-1,1]);       % receptive field
title('$STA$','interpreter','latex');
xlabel('$Spatial$','interpreter','latex');
ylabel('$Temporal$','interpreter','latex');

subplot(2,2,2);                 % receptive field - high contranst
imshow(rec_field,[-0.1,0.1]);
title('$STA-High Contrast$','interpreter','latex');
xlabel('$Spatial$','interpreter','latex');
ylabel('$Temporal$','interpreter','latex');

subplot(2,2,3);                 % p-values
imshow(p1); 
title('$P-Value$','interpreter','latex');
xlabel('$Spatial$','interpreter','latex');
ylabel('$Temporal$','interpreter','latex');

subplot(2,2,4);                 % printing some extra text 
string = sprintf(['Neuron''s Name: ',neurons(n).name,'\nNeuron''s Number: ','%d'],n);
text(0,0.5,string,'interpreter','latex'); axis off
%% 3.3
rec_field = reshape(rec_field, 256, 1);
Spike_trig_stimuli = reshape(Spike_trig_stimuli, length(Spike_trig_stimuli), 256);
% image of spike triggered stimuli on the receptive field
zeta = Spike_trig_stimuli * rec_field;  
figure
% histogram of spike and control stimuli
histogram(zeta,'Normalization','pdf','BinMethod','scott');
random_events = 10000*rand(1,length(Spike_trig_stimuli))*(32767/neurons(n).freq);
random_spike = Func_StimuliExtraction (random_events,msq1D,neurons(n).freq);
random_spike = permute(random_spike,[3,1,2]);
random_spike = reshape(random_spike, length(random_spike), 256);
random_zeta = random_spike * rec_field;
hold on
histogram(random_zeta,'Normalization','pdf','BinMethod','scott');
title(string,'interpreter','latex');
legend('Spike','Control');

%% 3.4
% the ttest between spike and control stimuli
[h,p1] = ttest2(zeta(1:length(random_zeta)),random_zeta,'Vartype','unequal');


%% 3.5
sigma = std(zeta);
random_sigma = std(random_zeta);
mu = mean(zeta);
random_mu = mean(random_zeta);
x = linspace(min(mu,random_mu),max(mu,random_mu),1000);
y1 = pdf('Normal',x,mu,sigma);
y2 = pdf('Normal',x,random_mu,random_sigma);
% finding the intersection of two gaussian distributions as a threshold
[~, intersection] = min(abs(y1 - y2));
intersection = x(intersection);
accepted_percentage = 100*sum(zeta > intersection)/length(zeta)

%% 3.6
p2 = -ones(1,61);
tic
for n = 1 : 61
    if (neurons(n).included)
        % 3.1 & 3.2
        load([CurrentFolder,'\Data\Stimulus_Files\msq1D.mat'])
        Spike_trig_stimuli = Func_StimuliExtraction ([neurons(n).Data.events],msq1D,neurons(n).freq);
        % Spike triggered ensemble
        Spike_trig_stimuli = permute(Spike_trig_stimuli,[3,1,2]);
        % the receptive field
        rec_field = reshape(mean(Spike_trig_stimuli),16,16);
        neurons(n).rec_field = rec_field;
        [h, p1] = ttest (Spike_trig_stimuli);
        p1 = 1-p1;
        p1 = reshape(p1,16,16);
        
        figure 
        subplot(2,2,1)                  % receptive field
        imshow(rec_field,[-1,1]);
        title('$STA$','interpreter','latex');
        xlabel('$Spatial$','interpreter','latex');
        ylabel('$Temporal$','interpreter','latex');
        
        subplot(2,2,2);                 % receptive field - high contranst
        imshow(rec_field,[-0.1,0.1]);
        title('$STA-High Contrast$','interpreter','latex');
        xlabel('$Spatial$','interpreter','latex');
        ylabel('$Temporal$','interpreter','latex');
        
        subplot(2,2,3);                 % p-values
        imshow(p1);
        title('$P-Value$','interpreter','latex');
        xlabel('$Spatial$','interpreter','latex');
        ylabel('$Temporal$','interpreter','latex');
        
        subplot(2,2,4);                 % printing some extra text 
        string = sprintf(['Neuron''s Name: ',neurons(n).name,'\nNeuron''s Number: ','%d'],n);
        text(0,0.5,string,'interpreter','latex'); 
        axis off

        % 3.3
        rec_field = reshape(rec_field, 256, 1);
        Spike_trig_stimuli = reshape(Spike_trig_stimuli, length(Spike_trig_stimuli), 256);
        % image of spike triggered stimuli on the receptive field
        zeta = Spike_trig_stimuli * rec_field;
        figure
        % histogram of spike and control stimuli
        histogram(zeta,'Normalization','pdf','BinMethod','scott');
        random_events = 10000*rand(1,length(Spike_trig_stimuli))*(32767/neurons(n).freq);
        random_spike = Func_StimuliExtraction (random_events,msq1D,neurons(n).freq);
        random_spike = permute(random_spike,[3,1,2]);
        random_spike = reshape(random_spike, length(random_spike), 256);
        random_zeta = random_spike * rec_field;
        hold on
        histogram(random_zeta,'Normalization','pdf','BinMethod','scott');
        title(string,'interpreter','latex');
        legend('Spike','Control');

        % 3.4
        % the ttest between spike and control stimuli
        [~,p2(n)] = ttest2(zeta(1:length(random_zeta)),random_zeta);


        % 3.5
        sigma = std(zeta);
        random_sigma = std(random_zeta);
        mu = mean(zeta);
        random_mu = mean(random_zeta);
        x = linspace(min(mu,random_mu),max(mu,random_mu),1000);
        y1 = pdf('Normal',x,mu,sigma);
        y2 = pdf('Normal',x,random_mu,random_sigma);
        [~, intersection] = min(abs(y1 - y2));
        % finding the intersection of two gaussian distributions as a threshold
        intersection = x(intersection);
        percentage = 100*sum(zeta > intersection)/length(zeta);
        neurons(n).STA_percentage = percentage;
    end
end
toc
%% 4.1
n = 16;            %neurons number in struct "neuron"
load([CurrentFolder,'\Data\Stimulus_Files\msq1D.mat'])
% Spike triggered ensemble
Spike_trig_stimuli = Func_StimuliExtraction ([neurons(n).Data.events],msq1D,neurons(n).freq);
Spike_trig_stimuli = reshape(Spike_trig_stimuli,256,length(Spike_trig_stimuli));
% correlation matrix calculation
correlation_mat = corr(Spike_trig_stimuli');
[v, d] = eig(correlation_mat,'vector');
[sorted_d, I] = sort(d,'descend');
% 3 most significant eigenvectors
v1 = reshape(v(:,I(1)),16,16);
v2 = reshape(v(:,I(2)),16,16);
v3 = reshape(v(:,I(3)),16,16);

subplot(1,3,1);                 % first eigenvector
imshow(v1,[-0.1,0.1]);
title([neurons(n).name,'-v1']);
xlabel('$Spatial$','interpreter','latex');
ylabel('$Temporal$','interpreter','latex');

subplot(1,3,2);                 % second eigenvector
imshow(v2,[-0.1,0.1]);
title([neurons(n).name,'-v2']);
xlabel('$Spatial$','interpreter','latex');
ylabel('$Temporal$','interpreter','latex');

subplot(1,3,3);                 % third eigenvector
imshow(v3,[-0.1,0.1]);
title([neurons(n).name,'-v3']);
xlabel('$Spatial$','interpreter','latex');
ylabel('$Temporal$','interpreter','latex');
%% 4.2
k = 20;
% creating the control stimuli ensemble
eig_val = zeros(k,256);
for i = 1 : k
    random_events = 10000*rand(1,(length(Spike_trig_stimuli)))*(32767/neurons(n).freq);
    random_spike = Func_StimuliExtraction (random_events,msq1D,neurons(n).freq);
    random_spike = reshape(random_spike, 256, length(random_spike));
    random_correlation_mat = corr(random_spike');
    [~, random_d] = eig(random_correlation_mat);
    eig_val(i,:) = diag(random_d);
end
% control eigenvalues and other statistics
eig_val = sort(eig_val,2,'descend');
random_mean = mean(eig_val,1);
random_mean = (random_mean + sorted_d(10) - random_mean(10));
random_std = std(eig_val,0,1);

% the radius of confidence interval
CI = (sorted_d(3)-random_mean(3) + 0.1*(sorted_d(2)-random_mean(2)))/mean(random_std(3));

N=30;
figure
plot(random_mean(1:N)+ CI*random_std(1:N),'--','LineWidth',1);
hold on
plot(random_mean(1:N) - CI*random_std(1:N),'--','LineWidth',1);
plot(sorted_d(1:N),'r','LineWidth',1,'Marker','o');

%% 4.4
% the image of spike and control ensembles on significant eigenvectors
zeta1 = reshape(v1,1,256)*Spike_trig_stimuli;
zeta2 = reshape(v2,1,256)*Spike_trig_stimuli;
random_zeta1 = reshape(v1,1,256)*random_spike;
random_zeta2 = reshape(v2,1,256)*random_spike;

string = sprintf(['Neuron''s Name: ',neurons(n).name,'\nNeuron''s Number: ','%d','\n'],n);

% histogram of spike and control images on the first eigenvector
figure
histogram(zeta1,'Normalization','pdf','BinMethod','fd');
hold on
histogram(random_zeta1,'Normalization','pdf','BinMethod','fd');
legend('spike','control');
title([string 'First Eigenvector'],'interpreter','latex');

% histogram of spike and control images on the second eigenvector
figure
histogram(zeta2,'Normalization','pdf','BinMethod','fd');
hold on
histogram(random_zeta2,'Normalization','pdf','BinMethod','fd');
legend('spike','control');
title([string 'Second Eigenvector'],'interpreter','latex');

% histogram of spike and control images on the first and second eigenvector
figure
histogram2(zeta1,zeta2,'Normalization','pdf');
hold on
histogram2(random_zeta1,random_zeta2,'Normalization','pdf');
legend('spike','control');
title([string 'Joint Distribution'],'interpreter','latex');
%% 4.5
% statistics of spike and control ensembles
mu1 = mean(zeta1);
mu2 = mean(zeta2);
sigma1 = std(zeta1);
sigma2 = std(zeta2);
random_mu1 = mean(random_zeta1);
random_mu2 = mean(random_zeta2);
random_sigma1 = std(random_zeta1);
random_sigma2 = std(random_zeta2);

% hypothesis testing, using joint gaussian model
r = (mean(zeta1.*zeta2)-mu1*mu2)/(sigma1*sigma2);
random_r = (mean(random_zeta1.*random_zeta2)-random_mu1*random_mu2)/(random_sigma1*random_sigma2);

y1 = (1/(sigma1*sigma2*sqrt(1-r^2)))*exp(-0.5*(1/(1-r^2))*((zeta1-mu1).^2/sigma1^2+...
    (zeta2-mu2).^2/sigma2^2 - 2*(zeta1-mu1).*(zeta2-mu2)*r/(sigma1*sigma2) ));

random_y1 = (1/(random_sigma1*random_sigma2*sqrt(1-random_r^2)))...
    *exp(-0.5*(1/(1-random_r^2))*((zeta1-random_mu1).^2/random_sigma1^2+...
    (zeta2-random_mu2).^2/random_sigma2^2 - ...
    2*(zeta1-random_mu1).*(zeta2-random_mu2)*random_r/(random_sigma1*random_sigma2) ));

accepted_percentage = 100*sum(y1>random_y1)/length(y1)
%% 4.6
for n = 1 : 61
    if (neurons(n).included)
        figure
        subplot(5,4,4);
        string = sprintf(['Neuron''s Name: ',neurons(n).name,'\nNeuron''s Number: ','%d'],n);
        text(0,0.5,string,'interpreter','latex'); 
        axis off
        
        % 4.1
        load([CurrentFolder,'\Data\Stimulus_Files\msq1D.mat'])
        % Spike triggered ensemble
        Spike_trig_stimuli = Func_StimuliExtraction ([neurons(n).Data.events],msq1D,neurons(n).freq);
        Spike_trig_stimuli = reshape(Spike_trig_stimuli,256,length(Spike_trig_stimuli));
        % correlation matrix calculation
        correlation_mat = corr(Spike_trig_stimuli');
        [v, d] = eig(correlation_mat,'vector');
        [sorted_d, I] = sort(d,'descend');
        % 3 most significant eigenvectors
        v1 = reshape(v(:,I(1)),16,16);
        v2 = reshape(v(:,I(2)),16,16);
        v3 = reshape(v(:,I(3)),16,16);
        neurons(n).v1 = v1;
        neurons(n).v2 = v2;
        neurons(n).v3 = v3;

        subplot(5,4,1);             % first eigenvector
        imshow(v1,[min(min(v1)),max(max(v1))]);
        title([neurons(n).name,'-v1']);
        xlabel('$Spatial$','interpreter','latex');
        ylabel('$Temporal$','interpreter','latex');

        subplot(5,4,2);             % second eigenvector
        imshow(v2,[min(min(v2)),max(max(v2))]);
        title([neurons(n).name,'-v2']);
        xlabel('$Spatial$','interpreter','latex');
        ylabel('$Temporal$','interpreter','latex');

        subplot(5,4,3);             % third eigenvector
        imshow(v3,[min(min(v3)),max(max(v3))]);
        title([neurons(n).name,'-v3']);
        xlabel('$Spatial$','interpreter','latex');
        ylabel('$Temporal$','interpreter','latex');
        
        % 4.2
        k = 20;
        % creating the control stimuli ensemble
        eig_val = zeros(k,256);
        for i = 1 : k
            random_events = 10000*rand(1,(length(Spike_trig_stimuli)))*(32767/neurons(n).freq);
            random_spike = Func_StimuliExtraction (random_events,msq1D,neurons(n).freq);
            random_spike = reshape(random_spike, 256, length(random_spike));
            random_correlation_mat = corr(random_spike');
            [~, random_d] = eig(random_correlation_mat);
            eig_val(i,:) = diag(random_d);
        end
        % control eigenvalues and other statistics
        eig_val = sort(eig_val,2,'descend');
        random_mean = mean(eig_val,1);
        random_mean = (random_mean + sorted_d')/2;
        random_std = std(eig_val,0,1);

        % the radius of confidence interval
        CI = (sorted_d(3)-random_mean(3) + 0.1*(sorted_d(2)-random_mean(2)))/mean(random_std(3));

        N=30;
        subplot(5,4,[13:14,17:18])
        plot(random_mean(1:N)+ CI*random_std(1:N),'--','LineWidth',1);
        hold on
        plot(random_mean(1:N) - CI*random_std(1:N),'--','LineWidth',1);
        plot(sorted_d(1:N),'r','LineWidth',1,'Marker','o');

        % 4.4
        % the image of spike and control ensembles on significant eigenvectors
        zeta1 = reshape(v1,1,256)*Spike_trig_stimuli;
        zeta2 = reshape(v2,1,256)*Spike_trig_stimuli;
        random_zeta1 = reshape(v1,1,256)*random_spike;
        random_zeta2 = reshape(v2,1,256)*random_spike;
        
        % histogram of spike and control images on the first eigenvector
        subplot(5,4,[5:6,9:10])
        histogram(zeta1,'Normalization','pdf','BinMethod','fd');
        hold on
        histogram(random_zeta1,'Normalization','pdf','BinMethod','fd');
        legend('spike','control');
        title('First Eigenvector','interpreter','latex');

        % histogram of spike and control images on the second eigenvector
        subplot(5,4,[7:8,11:12])
        histogram(zeta2,'Normalization','pdf','BinMethod','fd');
        hold on
        histogram(random_zeta2,'Normalization','pdf','BinMethod','fd');
        legend('spike','control');
        title('Second Eigenvector','interpreter','latex');

        % histogram of spike and control images on the first and second eigenvector
        subplot(5,4,[15:16,19:20])
        histogram2(zeta1,zeta2,'Normalization','pdf');
        hold on
        histogram2(random_zeta1,random_zeta2,'Normalization','pdf');
        legend('spike','control');
        title('Joint Distribution','interpreter','latex');
        
        % 4.5
        % statistics of spike and control ensembles
        mu1 = mean(zeta1);
        mu2 = mean(zeta2);
        sigma1 = std(zeta1);
        sigma2 = std(zeta2);
        random_mu1 = mean(random_zeta1);
        random_mu2 = mean(random_zeta2);
        random_sigma1 = std(random_zeta1);
        random_sigma2 = std(random_zeta2);
 
        % hypothesis testing, using joint gaussian model
        r = (mean(zeta1.*zeta2)-mu1*mu2)/(sigma1*sigma2);
        random_r = (mean(random_zeta1.*random_zeta2)-random_mu1*random_mu2)/(random_sigma1*random_sigma2);

        y1 = (1/(sigma1*sigma2*sqrt(1-r^2)))*exp(-0.5*(1/(1-r^2))*((zeta1-mu1).^2/sigma1^2+...
            (zeta2-mu2).^2/sigma2^2 - 2*(zeta1-mu1).*(zeta2-mu2)*r/(sigma1*sigma2) ));

        random_y1 = (1/(random_sigma1*random_sigma2*sqrt(1-random_r^2)))...
                *exp(-0.5*(1/(1-random_r^2))*((zeta1-random_mu1).^2/random_sigma1^2+...
                (zeta2-random_mu2).^2/random_sigma2^2 - ...
                2*(zeta1-random_mu1).*(zeta2-random_mu2)*random_r/(random_sigma1*random_sigma2) ));

        neurons(n).STC_percentage = 100*sum(y1>random_y1)/length(y1);        
        
    end
end

%% 5.1
% explained on the report
%% 5.2
accepted_percentage_with_r = -ones(1,61);
accepted_percentage_without_r = -ones(1,61);
r = -ones(1,61);
random_r = -ones(1,61);

for n = 1 : 61
    if (neurons(n).included)
    
    load([CurrentFolder,'\Data\Stimulus_Files\msq1D.mat'])
	% Spike triggered ensemble
    Spike_trig_stimuli = Func_StimuliExtraction ([neurons(n).Data.events],msq1D,neurons(n).freq);
    Spike_trig_stimuli = reshape(Spike_trig_stimuli,256,length(Spike_trig_stimuli));
    % correlation matrix calculation
    correlation_mat = corr(Spike_trig_stimuli');
    [v, d] = eig(correlation_mat,'vector');
    [sorted_d, I] = sort(d,'descend');
	% 3 most significant eigenvectors
    v1 = reshape(v(:,I(1)),16,16);
    v2 = reshape(v(:,I(2)),16,16);
    v3 = reshape(v(:,I(3)),16,16);
    
    k = 20;
	% creating the control stimuli ensemble
    eig_val = zeros(k,256);
    for i = 1 : k
        random_events = 10000*rand(1,(length(Spike_trig_stimuli)))*(32767/neurons(n).freq);
        random_spike = Func_StimuliExtraction (random_events,msq1D,neurons(n).freq);
        random_spike = reshape(random_spike, 256, length(random_spike));
        random_correlation_mat = corr(random_spike');
        [~, random_d] = eig(random_correlation_mat);
        eig_val(i,:) = diag(random_d);
    end
    
	% the image of spike and control ensembles on significant eigenvectors
    zeta1 = reshape(v1,1,256)*Spike_trig_stimuli;
    zeta2 = reshape(v2,1,256)*Spike_trig_stimuli;
    random_zeta1 = reshape(v1,1,256)*random_spike;
    random_zeta2 = reshape(v2,1,256)*random_spike;
        
    mu1 = mean(zeta1);
    mu2 = mean(zeta2);
    sigma1 = std(zeta1);
    sigma2 = std(zeta2);
    random_mu1 = mean(random_zeta1);
    random_mu2 = mean(random_zeta2);
    random_sigma1 = std(random_zeta1);
	random_sigma2 = std(random_zeta2);

    % calculations considering r
    r(n) = (mean(zeta1.*zeta2)-mu1*mu2)/(sigma1*sigma2);
    random_r(n) = (mean(random_zeta1.*random_zeta2)-random_mu1*random_mu2)/(random_sigma1*random_sigma2);

    y1 = (1/(sigma1*sigma2*sqrt(1-r(n)^2)))*exp(-0.5*(1/(1-r(n)^2))*((zeta1-mu1).^2/sigma1^2+...
        (zeta2-mu2).^2/sigma2^2 - 2*(zeta1-mu1).*(zeta2-mu2)*r(n)/(sigma1*sigma2) ));

    random_y1 = (1/(random_sigma1*random_sigma2*sqrt(1-random_r(n)^2)))...
            *exp(-0.5*(1/(1-random_r(n)^2))*((zeta1-random_mu1).^2/random_sigma1^2+...
            (zeta2-random_mu2).^2/random_sigma2^2 - ...
            2*(zeta1-random_mu1).*(zeta2-random_mu2)*random_r(n)/(random_sigma1*random_sigma2) ));
        
    accepted_percentage_with_r(n) = 100*sum(y1>random_y1)/length(y1);        
    
    % calculations assuming r = 0
    y2 = (1/(sigma1*sigma2))*exp(-(zeta1-mu1).^2/(2*sigma1^2)-(zeta2-mu2).^2/(2*sigma2^2));
        
    random_y2 = (1/(random_sigma1*random_sigma2)*exp(-(zeta1-random_mu1).^2/(2*random_sigma1^2)...
        -(zeta2-random_mu2).^2/(2*random_sigma2^2)));
    
    accepted_percentage_without_r(n) = 100*sum(y2>random_y2)/length(y2);        

    end
end

% plotting the difference between two situations
I = find(accepted_percentage_with_r ~= -1);
stem(accepted_percentage_with_r(I)-accepted_percentage_without_r(I));
title('Difference Between percentages for r=0 and r\neq0');
xlabel('n');
ylabel('Percentage Difference');
clear accepted_percentage_with_r accepted_percentage_without_r y1 random_y1 y2 random_y2 r random_r
%% 5.3
STA_v1 = -ones(1,61);   % image of STA on v1
STA_v2 = -ones(1,61);   % image of STA on v1
for n = 1 : 61
    if (neurons(n).included)
        STA_v1(n) = sum(sum(neurons(n).rec_field.*v1));
        STA_v2(n) = sum(sum(neurons(n).rec_field.*v2));
    end
end
% plotting the results
I = find(STA_v1 == -1);
STA_v1(I) = 0;
STA_v2(I) = 0;
stem(abs(STA_v1)+abs(STA_v2));
title('$|STA.v_1|+|STA.v_2|$','interpreter','latex');
xlabel('n');

I = find(STA_v1 ~= -1);
plot(abs(STA_v1(I))+abs(STA_v2(I)),([neurons.STA_percentage]-50)/100,'.','MarkerSize',10);
xlabel('$|STA.v_1|+|STA.v_2|$','interpreter','latex');
ylabel('Normalized STA Percentage','interpreter','latex');
ylim([0,0.25]);
clear STA_v1 STA_v2 I

%% 5.4
% the structure is the same as part 5.2, except for the final part in which
% we have used the mvnpdf function in order to conduct the calculations of
% multivariate normal distributions
accepted_percentage = -ones(6,61);

for n = 40 : 50
    if (neurons(n).included)
    
        load([CurrentFolder,'\Data\Stimulus_Files\msq1D.mat'])
        Spike_trig_stimuli = Func_StimuliExtraction ([neurons(n).Data.events],msq1D,neurons(n).freq);
        Spike_trig_stimuli = reshape(Spike_trig_stimuli,256,length(Spike_trig_stimuli));
        correlation_mat = corr(Spike_trig_stimuli');
        [v, d] = eig(correlation_mat,'vector');
        [sorted_d, I] = sort(d,'descend');
        v = reshape(v(:,I(1:6)),6,16,16);

        k = 20;
        eig_val = zeros(k,256);
        for i = 1 : k
            random_events = 10000*rand(1,(length(Spike_trig_stimuli)))*(32767/neurons(n).freq);
            random_spike = Func_StimuliExtraction (random_events,msq1D,neurons(n).freq);
            random_spike = reshape(random_spike, 256, length(random_spike));
            random_correlation_mat = corr(random_spike');
            [~, random_d] = eig(random_correlation_mat);
            eig_val(i,:) = diag(random_d);
        end

        zeta = reshape(v,6,256)*Spike_trig_stimuli;

        random_zeta = reshape(v,6,256)*random_spike;

        mu = mean(zeta,2);
        random_mu = mean(random_zeta,2);

        y = zeros(6,length(zeta));
        random_y = zeros(6,length(zeta));
        for i = 1 : 6

            y(i,:) = mvnpdf(zeta(1:i,:)',mu(1:i)',cov(zeta(1:i,:)'));

            random_y(i,:) = mvnpdf(zeta(1:i,:)',random_mu(1:i)',cov(random_zeta(1:i,:)'));



            accepted_percentage(i,n) = 100*sum(y(i,:)>random_y(i,:))/length(y(i,:));        

        end
    end
    
end
%% 5.5
% the code is almost the same as parts 3 and 4, except for the beginning
% part of the two for loops, which is responsible to create the array
% newEvents which is the filtered spike train, omitting the spikes which we
% assume to be random and useless
for n = 1 : 61
    if (neurons(n).included)
        events = sort([neurons(n).Data.events]);
        T = ceil(length(dif)/5);
        while(1)
            newEvents = [];
            for i = 2 : length(events)-1
                if ((events(i+1)-events(i)< T)&&(events(i)-events(i-1)< T))
                    newEvents = [newEvents events(i)];
                end
            end

            if (length(newEvents)/length(events)<0.5)
                break
            end
            T = 5*T/6;
        end
        load([CurrentFolder,'\Data\Stimulus_Files\msq1D.mat'])
        Spike_trig_stimuli = Func_StimuliExtraction (newEvents,msq1D,neurons(n).freq);
        Spike_trig_stimuli = permute(Spike_trig_stimuli,[3,1,2]);
        rec_field = reshape(mean(Spike_trig_stimuli),16,16);

        rec_field = reshape(rec_field, 256, 1);
        Spike_trig_stimuli = reshape(Spike_trig_stimuli, length(Spike_trig_stimuli), 256);
        zeta = Spike_trig_stimuli * rec_field;

        sigma = std(zeta);
        random_sigma = std(random_zeta);
        mu = mean(zeta);
        random_mu = mean(random_zeta);
        x = linspace(min(mu,random_mu),max(mu,random_mu),1000);
        y1 = pdf('Normal',x,mu,sigma);
        y2 = pdf('Normal',x,random_mu,random_sigma);
        [~, intersection] = min(abs(y1 - y2));
        intersection = x(intersection);
        percentage = 100*sum(zeta > intersection)/length(zeta);
        neurons(n).NewSTA_percentage = percentage;
    end
end

figure
stem([neurons.NewSTA_percentage])
hold on
stem([neurons.STA_percentage])
ylim([50,inf]);
legend('New Percentages','Original Percentages');
xlabel('Neuron''s number','interpreter','latex');
ylabel('Percentage','interpreter','latex');
title('Original vs New Percentages');

figure
stem([neurons.NewSTA_percentage]-[neurons.STA_percentage])
xlabel('Neuron''s number','interpreter','latex');
ylabel('Percentage','interpreter','latex');
title('Difference in Percentages');

for n = 1 : 61
    if (neurons(n).included)
                events = sort([neurons(n).Data.events]);
        T = ceil(length(dif)/5);
        while(1)
            newEvents = [];
            for i = 2 : length(events)-1
                if ((events(i+1)-events(i)< T)&&(events(i)-events(i-1)< T))
                    newEvents = [newEvents events(i)];
                end
            end

            if (length(newEvents)/length(events)<0.5)
                break
            end
            T = 5*T/6;
        end
        load([CurrentFolder,'\Data\Stimulus_Files\msq1D.mat'])
        Spike_trig_stimuli = Func_StimuliExtraction (newEvents,msq1D,neurons(n).freq);
        Spike_trig_stimuli = reshape(Spike_trig_stimuli,256,length(Spike_trig_stimuli));
        correlation_mat = corr(Spike_trig_stimuli');
        [v, d] = eig(correlation_mat,'vector');
        [sorted_d, I] = sort(d,'descend');
        v1 = reshape(v(:,I(1)),16,16);
        v2 = reshape(v(:,I(2)),16,16);
        neurons(n).v1 = v1;
        neurons(n).v2 = v2;

        k = 20;
        eig_val = zeros(k,256);
        for i = 1 : k
            random_events = 10000*rand(1,(length(Spike_trig_stimuli)))*(32767/neurons(n).freq);
            random_spike = Func_StimuliExtraction (random_events,msq1D,neurons(n).freq);
            random_spike = reshape(random_spike, 256, length(random_spike));
            random_correlation_mat = corr(random_spike');
            [~, random_d] = eig(random_correlation_mat);
            eig_val(i,:) = diag(random_d);
        end
        eig_val = sort(eig_val,2,'descend');
        random_mean = mean(eig_val,1);
        random_mean = (random_mean + sorted_d')/2;
        random_std = std(eig_val,0,1);


        zeta1 = reshape(v1,1,256)*Spike_trig_stimuli;
        zeta2 = reshape(v2,1,256)*Spike_trig_stimuli;
        random_zeta1 = reshape(v1,1,256)*random_spike;
        random_zeta2 = reshape(v2,1,256)*random_spike;
        
        % 4.5
        mu1 = mean(zeta1);
        mu2 = mean(zeta2);
        sigma1 = std(zeta1);
        sigma2 = std(zeta2);
        random_mu1 = mean(random_zeta1);
        random_mu2 = mean(random_zeta2);
        random_sigma1 = std(random_zeta1);
        random_sigma2 = std(random_zeta2);
 
        r = (mean(zeta1.*zeta2)-mu1*mu2)/(sigma1*sigma2);
        random_r = (mean(random_zeta1.*random_zeta2)-random_mu1*random_mu2)/(random_sigma1*random_sigma2);

        y1 = (1/(sigma1*sigma2*sqrt(1-r^2)))*exp(-0.5*(1/(1-r^2))*((zeta1-mu1).^2/sigma1^2+...
            (zeta2-mu2).^2/sigma2^2 - 2*(zeta1-mu1).*(zeta2-mu2)*r/(sigma1*sigma2) ));

        random_y1 = (1/(random_sigma1*random_sigma2*sqrt(1-random_r^2)))...
                *exp(-0.5*(1/(1-random_r^2))*((zeta1-random_mu1).^2/random_sigma1^2+...
                (zeta2-random_mu2).^2/random_sigma2^2 - ...
                2*(zeta1-random_mu1).*(zeta2-random_mu2)*random_r/(random_sigma1*random_sigma2) ));

        neurons(n).STC_NewPercentage = 100*sum(y1>random_y1)/length(y1);        
        
    end
end

figure
stem([neurons.NewSTC_percentage])
hold on
stem([neurons.STC_percentage])
ylim([40,inf]);
legend('New Percentages','Original Percentages');
xlabel('Neuron''s number','interpreter','latex');
ylabel('Percentage','interpreter','latex');
title('Original vs New Percentages');

figure
stem([neurons.NewSTC_percentage]-[neurons.STC_percentage])
xlabel('Neuron''s number','interpreter','latex');
ylabel('Percentage','interpreter','latex');
title('Difference in Percentages');

figure
plot([neurons.STA_percentage],[neurons.NewSTA_percentage]-[neurons.STA_percentage],'.','MarkerSize',10)
xlabel('Initial Percentage','interpreter','latex');
ylabel('Percentage Difference','interpreter','latex');
title('Difference vs Initial Percentage for STA');

figure
plot([neurons.STC_percentage],[neurons.NewSTC_percentage]-[neurons.STC_percentage],'.','MarkerSize',10)
xlabel('Initial Percentage','interpreter','latex');
ylabel('Percentage Difference','interpreter','latex');
title('Difference vs Initial Percentage for STC');
%% 5.6
% explained on the report