function [sfmCos] = semDistSRModel()

words = {'SPIDER', 'LIQUID', 'DIAMOND' 'IRON', 'BUBBLE', 'MOMENT', 'SUBJECT' , 'RESEARCH', 'FINGER' , 'BUTTON', 'SUCCESS', 'FAILURE'};


sfmCos = zeros(length(words),length(words));

try 
    load('cosDistSR.mat');
    
catch
    
    pairs = nchoosek(1:length(words),2);
    allCos = zeros(length(words),length(words));

    for p = 1:size(pairs,1)

        w1 = words{1,pairs(p,1)};
        w2 = words{1,pairs(p,2)};

        semDist = semDistance(w1,w2);
        allCos(pairs(p,1),pairs(p,2)) = semDist;
        allCos(pairs(p,2),pairs(p,1)) = semDist;
        clear semDist
    end

end


for d = 1:size(allCos,1)
    allTr = allCos(d,:);
    allTr(d) = [];
    sfTr = softmax(allTr');
    sfTr = [sfTr(1:d-1); 0 ; sfTr(d:end)];
    sfmCos(d,:) = sfTr';
end

end


