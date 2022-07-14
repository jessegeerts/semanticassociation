function[cs] = semDistance(word1,word2)

emb = fastTextWordEmbedding;

vector1 = word2vec(emb,word1);
vector2 = word2vec(emb,word2);
cs = getCosineSimilarity(vector1,vector2);
% ed = sqrt(sum((vector1 - vector2) .^ 2));
% dp = dot(vector1,vector2);

