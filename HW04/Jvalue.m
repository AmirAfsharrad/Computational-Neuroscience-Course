function [J] = Jvalue(Feature,a,b)
    
    Xfeature = Feature(Feature(:,end-1)==a , 1:end-2);

    Yfeature = Feature(Feature(:,end-1)==b , 1:end-2);
    
% 	Feature = [Xfeature;Yfeature];
            
    mu0 = mean([Xfeature;Yfeature],1);
    mu1 = mean(Xfeature,1);
    mu2 = mean(Yfeature,1);
    sigma1 = var(Xfeature,0,1);
    sigma2 = var(Yfeature,0, 1);
    
    J = (abs(mu0 - mu1).^2 + abs(mu0 - mu2).^2)./(sigma1 + sigma2+eps);
   

