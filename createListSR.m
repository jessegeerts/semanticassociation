%Criteria to create list: semantic distance as cos in word2vec needs to be
%spread over a broad range of values; avoid close spelling similarity



U = readtable('iEEGFreeRecallNounPool.xlsx');
U = table2cell(U);

T = readtable('NounPool.csv');
T = table2cell(T);

T = [T; U];
a = randperm(length(T));
T = T(a);
emb = fastTextWordEmbedding;

%%

% for l = 1:2
    
list = [];

%% find list of frequent words

while isempty(list)
    
    lst = T(randsample(length(T),30));
    % write words in lower case only
    lst = lower(lst);
    % find embedding vectors in word2vec for all the words in the list
    V = word2vec(emb,lst);
    
    %%
    
    allWords = nchoosek(1:length(lst),2);

    for w = 1:size(allWords,1)
        w1= allWords(w,1);
        w2 = allWords(w,2); 
        wd = lst(w1);
        wd2 = lst(w2);

        % find spelling distance between each pair of words
        ortD(w1,w2) = editDistance(wd,wd2); 
        semD(w1,w2) =  getCosineSimilarity(V(w1,:),V(w2,:));
        tempD(w1,w2) = abs(w1-w2);
            
    end
    
    semDist = semD(:);
    semDist(semDist == 0) = [];
    ortDist = ortD(:);
    ortDist(ortDist == 0) = [];
    
    minSDist = min(unique(semDist));
    maxSDist = max(unique(semDist));
    
    h = kstest(semDist);
    
    if min(ortDist)>1 & mean(ortDist)>5 & maxSDist-minSDist>0.5 & h==0
        list = lst;
    end
        clear ortD
        %clear B lst coho V coeff score latent tsquared explained mu

   
% end

list = lst;

save(['list'  num2str(l) '_SR.mat'],'list', 'semDist', 'semD', 'ortDist', 'ortD')


end

%% Check order of words in list

for w = 1:size(allWords,1)
        w1= allWords(w,1);
        w2 = allWords(w,2); 
        
        % find spelling distance between each pair of words
        
        tempD(w1,w2) = abs(w1-w2);
       
            
end
    

tempDist = tempD(:);
tempDist(tempDist ==0) = [];

scatter(tempDist,semDist)
[coef, p] = corr([tempDist,semDist]);

f=fit(tempDist,semDist,'poly1');
plot(f,tempDist,semDist,'o')
ylabel('cosSemDist')
xlabel('tempDist')