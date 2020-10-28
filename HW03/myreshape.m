function y = myreshape(x)
    y = [squeeze(x(:,2,:)),squeeze(x(:,3,:)),squeeze(x(:,4,:)),squeeze(x(:,5,:)),...
        squeeze(x(:,6,:)),squeeze(x(:,7,:)),squeeze(x(:,8,:)),squeeze(x(:,9,:))];
end