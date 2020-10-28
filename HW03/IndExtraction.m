function [output] = IndExtraction(subject)
    flash = subject.train(10,:)~=0;
    flash = flash - [0, flash(1:end-1)];
    flash(flash==-1) = 0;
    target = subject.train(11,:)~=0;
    non_target = flash & (~target);
    target = flash & target;
    num = 1 : length(target);
    target = target.*num;
    non_target = non_target.*num;

    output.Train_target = target(target~=0);
    output.Train_nontarget = non_target(non_target~=0);

	flash = subject.test(10,:)~=0;
    flash = flash - [0, flash(1:end-1)];
    flash(flash==-1) = 0;
    target = subject.test(11,:)~=0;
    non_target = flash & (~target);
    target = flash & target;

    num = 1 : length(target);
    target = target.*num;
    non_target = non_target.*num;

    output.Test_target = target(target~=0);
    output.Test_nontarget = non_target(non_target~=0);


end

