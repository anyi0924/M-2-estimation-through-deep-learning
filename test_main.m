close all
clear all
 mySetup();
load ('save/pattern_vgg/FT-net-30-trained','net');
net = dagnn.DagNN.loadobj(net) ;
  net.move('gpu');
net.mode = 'test';
load test_data %here the test_data is from 3-mode case.
images(:,:,1,:)=test_pattern;
n=size(test_pattern,3);
for ii = 1:1:n
    im = single(images(:,:,1,ii:ii)) ;    
  im = gpuArray(im);
    % run the CNN
    predVar = net.getVarIndex('prediction');
    inputVar = 'input' ;
    net.eval({inputVar, im});
    tmp = squeeze(double(gather(net.vars(predVar).value))) ;
    neto(1:2,ii:ii) = double(tmp(1:2,:));
end
output=3*neto;
