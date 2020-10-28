function [Word] = WordRecognizer(Subject,N,Fs,string, TestPredIndex)

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
            score_table = zeros(6,6);
            for j = 1 : 12
                L{i,j} = Subject(N).test(10,I(TestPredIndex((TestPredIndex>((i-1)*180+(j-1)*15)) &...
                    (TestPredIndex<(i-1)*180+j*15+1))));
                if (~isnan(mode(L{i,j}(L{i,j}<7))))
                [~,~,a] = mode(L{i,j}(L{i,j}<7))
                Lc = a{1}
                                score_table(:,Lc) = score_table(:,Lc)+1;

                end
                if(~isnan(mode(L{i,j}(L{i,j}>6))))
                [~,~,b] = mode(L{i,j}(L{i,j}>6))
                Lr = b{1}
                score_table(Lr-6,:) = score_table(Lr-6,:)+1;
                end
            end
            if(sum(sum(score_table==max(max(score_table))))>1)
                for k = 1 : 36
                    if(score_table(k)==max(max(score_table)))
                        score_table = zeros(6,6);
                        score_table(k) = 1;
%                         break
                    end
                end
            end
            
            Word(i) = lookup(score_table == max(max(score_table)));
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