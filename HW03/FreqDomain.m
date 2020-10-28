function [t, State, X1] = FreqDomain(Subject)
    for i = 1 : size(Subject.train_target,1)
        for j = 2 : 9
            alpha_train_target(i,j-1,:) = filter(BPF(1001,8,15,64),1,Subject.train_target(i,j,:));
            beta_train_target(i,j-1,:) = filter(BPF(1001,15,31,64),1,Subject.train_target(i,j,:));
            alpha_test_target(i,j-1,:) = filter(BPF(1001,8,15,64),1,Subject.test_target(i,j,:));
            beta_test_target(i,j-1,:) = filter(BPF(1001,15,31,64),1,Subject.test_target(i,j,:));
        end
    end
    

    for i = 1 : size(Subject.train_nontarget,1)
        for j = 2 : 9
            alpha_train_nontarget(i,j-1,:) = filter(BPF(1001,8,15,64),1,Subject.train_target(i,j,:));
            beta_train_nontarget(i,j-1,:) = filter(BPF(1001,15,31,64),1,Subject.train_target(i,j,:));
            alpha_test_nontarget(i,j-1,:) = filter(BPF(1001,8,15,64),1,Subject.test_target(i,j,:));
            beta_test_nontarget(i,j-1,:) = filter(BPF(1001,15,31,64),1,Subject.test_target(i,j,:));
        end
    end
    
    L = floor (size(Subject.train_target,3)/5);
    for i = 1 : size(Subject.train_target,1)
        for j = 1 : 8
            for k = 1 : L
        X1(i,j,k) = norm(alpha_train_target(i,j,(k-1)*L+1:k*L))^2;
            end
            for k = 1 : L
                X1(i,j,k+L) = norm(beta_train_target(i,j,(k-1)*L+1:k*L))^2;
            end
        end
    end
        

    for i = 1 : L
        X1(i,1) = norm(delta1(1 + 1000*(i-1):1000*i))^2/1000;
        X1(i,2) = norm(theta1(1 + 1000*(i-1):1000*i))^2/1000;
        X1(i,3) = norm(alpha1(1 + 1000*(i-1):1000*i))^2/1000;
        X1(i,4) = norm(beta1(1 + 1000*(i-1):1000*i))^2/1000;
        X1(i,5) = norm(delta2(1 + 1000*(i-1):1000*i))^2/1000;
        X1(i,6) = norm(theta2(1 + 1000*(i-1):1000*i))^2/1000;
        X1(i,7) = norm(alpha2(1 + 1000*(i-1):1000*i))^2/1000;
        X1(i,8) = norm(beta2(1 + 1000*(i-1):1000*i))^2/1000;
        X1(i,9) = norm(data(3, 1 + 1000*(i-1) : 1000*i))^2/1000;
        X1(i,10) = norm(data(4, 1 + 1000*(i-1) : 1000*i))^2/1000;
    end

end