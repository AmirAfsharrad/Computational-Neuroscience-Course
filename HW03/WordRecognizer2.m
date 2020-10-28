function [Word] = WordRecognizer2(Subject,N,Fs,string, TestPredIndex)

[feature, I] = FeatureExtraction(Subject(N),Fs,string);

if (strcmp(Subject(N).method,'RC'))
    lookup = ['A','B','C','D','E','F';...
        'G','H','I','J','K','L';...
        'M','N','O','P','Q','R';...
        'S','T','U','V','W','X';...
        'Y','Z','0','1','2','3';...
        '4','5','6','7','8','9'];
    
    if (strcmp(string,'train'))
        if(nargin == 4)
            LA = Subject(N).train(11,Subject(N).train(10,:)~=0 )==1;
            TestPredIndex = find(LA == 1);
        end
        
        for i = 1 : 5
            L{i} = Subject(N).train(10,I(TestPredIndex((TestPredIndex>(i-1)*180) & (TestPredIndex<i*180+1))));
            Lc(i) = mode(L{i}(L{i}<7));
            Lr(i) = mode(L{i}(L{i}>6));
            Word(i) = lookup(Lr(i)-6,Lc(i));
        end
    end
    
    if (strcmp(string,'test'))
        if(nargin == 4)
            LA = Subject(N).test(11,Subject(N).test(10,:)~=0 )==1;
            TestPredIndex = find(LA == 1);
        end
        
        
        for i = 1 : 5
            L{i} = Subject(N).test(10,I(TestPredIndex((TestPredIndex>(i-1)*180) & (TestPredIndex<i*180+1))));
            Lc(i) = mode(L{i}(L{i}<7));
            Lr(i) = mode(L{i}(L{i}>6));
            Word(i) = lookup(Lr(i)-6,Lc(i));
        end
            


    end
end

if(strcmp(Subject(N).method,'SC'))
    lookup = ['A','B','C','D','E','F',...
        'G','H','I','J','K','L',...
        'M','N','O','P','Q','R',...
        'S','T','U','V','W','X',...
        'Y','Z','0','1','2','3',...
        '4','5','6','7','8','9'];
    
    if (strcmp(string,'train'))
        if(nargin == 4)
            LA = Subject(N).train(11,Subject(N).train(10,:)~=0 )==1;
            TestPredIndex = find(LA == 1);
        end
        
        for i = 1 : 5
            L{i} = Subject(N).train(10,I(TestPredIndex((TestPredIndex>(i-1)*540) & (TestPredIndex<i*540+1))));
            Lnum(i) = mode(L{i});
            Word(i) = lookup(Lnum(i));
        end
    end
    
    if (strcmp(string,'test'))
        if(nargin == 4)
            LA = Subject(N).test(11,Subject(N).test(10,:)~=0 )==1;
            TestPredIndex = find(LA == 1);
        end
        
        
        for i = 1 : 5
            L{i} = Subject(N).test(10,I(TestPredIndex((TestPredIndex>(i-1)*540) & (TestPredIndex<i*540+1))));
            Lnum(i) = mode(L{i});
            Word(i) = lookup(Lnum(i));
        end
    end
end


end