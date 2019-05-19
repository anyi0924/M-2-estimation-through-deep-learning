function setLearningRate( net )

%imNet
for ii = 1:30
    net.params(ii).learningRate = 0.01 ;
    net.params(ii).weightDecay = 0.0005 ;
end




