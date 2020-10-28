function [t, State, X] = FeatureExtraction(edf_path, hyp_path)
    TimeState = AnnotExtract(hyp_path);
    t = 0:10:TimeState(1,end);
    State = zeros(1,TimeState(1,end)/10);
    State(TimeState(1,:)/10+1) = TimeState(2,:)+1;
    for i = 2 : length(State)
        if(State(i) == 0)
            State(i) = State(i-1);
        end
    end
    State = State-1;
    State = State';
    
    [~, data] = edfread(edf_path);

    alpha1 = filter(BPF(1001,8,15,100),1,data(1,:));
    alpha2 = filter(BPF(1001,8,15,100),1,data(2,:));
    beta1 = filter(BPF(1001,16,31,100),1,data(1,:));
    beta2 = filter(BPF(1001,16,31,100),1,data(2,:));
    theta1 = filter(BPF(1001,4,7,100),1,data(1,:));
    theta2 = filter(BPF(1001,4,7,100),1,data(2,:));
    delta1 = filter(BPF(1001,0.5,4,100),1,data(1,:));
    delta2 = filter(BPF(1001,0.5,4,100),1,data(2,:));

    count = size(data,2);
    while(data(1,count)==data(1,count-1))
        count = count-1;
    end

    L = min(floor(count/1000),length(t));

	X = zeros(L, 10);

    for i = 1 : L
        X(i,1) = norm(delta1(1 + 1000*(i-1):1000*i))^2/1000;
        X(i,2) = norm(theta1(1 + 1000*(i-1):1000*i))^2/1000;
        X(i,3) = norm(alpha1(1 + 1000*(i-1):1000*i))^2/1000;
        X(i,4) = norm(beta1(1 + 1000*(i-1):1000*i))^2/1000;
        X(i,5) = norm(delta2(1 + 1000*(i-1):1000*i))^2/1000;
        X(i,6) = norm(theta2(1 + 1000*(i-1):1000*i))^2/1000;
        X(i,7) = norm(alpha2(1 + 1000*(i-1):1000*i))^2/1000;
        X(i,8) = norm(beta2(1 + 1000*(i-1):1000*i))^2/1000;
        X(i,9) = norm(data(3, 1 + 1000*(i-1) : 1000*i))^2/1000;
        X(i,10) = norm(data(4, 1 + 1000*(i-1) : 1000*i))^2/1000;
    end

end