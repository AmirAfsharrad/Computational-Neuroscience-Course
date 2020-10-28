function Output = Func_StimuliExtraction(events,msq1D,frequency)
    N = length(events);
    T = 10000/frequency;
%     Output = zeros(16,16,N);
    j = 1;
    for i = 1 : N
        n = ceil(events(i)/T);
        if (n>16)
            Output(:,:,j) = msq1D((n-15):n , :);
            j = j+1;
        end
    end
end