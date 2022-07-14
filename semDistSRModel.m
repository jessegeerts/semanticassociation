words = {'SPIDER', 'LIQUID', 'DIAMOND' 'IRON', 'BUBBLE', 'MOMENT', 'SUBJECT' , 'RESEARCH', 'FINGER' , 'BUTTON', 'SUCCESS', 'FAILURE'};
% 
% pairs = nchoosek(1:length(words),2);
% allCos = zeros(length(words),length(words));
sfmCos = zeros(length(words),length(words));

% for p = 1:11%size(pairs,1)
%     
%     w1 = words{1,pairs(p,1)};
%     w2 = words{1,pairs(p,2)};
%     
%     semDist = semDistance(w1,w2);
%     allCos(pairs(p,1),pairs(p,2)) = semDist;
%     allCos(pairs(p,2),pairs(p,1)) = semDist;
%     clear semDist
% end


 cosDist = tril(allCos);
 cosDist = cosDist(:);
 cosDist(cosDist ==0) =[];
 sfMCos = softmax(cosDist);
 
 for c = 1:length(cosDist)
     
     cs = cosDist(c);
     [rCs, cCs] = find(allCos == cs);
     sfmCos(rCs(1),cCs(1)) = sfMCos(c);
      sfmCos(rCs(2),cCs(2)) = sfMCos(c);
     
 end
     
 
 cosDist = softmax(cosDist);
 
 
