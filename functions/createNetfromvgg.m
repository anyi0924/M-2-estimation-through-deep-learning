function net = createNetfromvgg()

vggnet = load('imagenet-vgg-verydeep-16.mat') ; %This is the orginal VGG-16 model, which can be acquired from http://www.vlfeat.org/matconvnet/
net.layers = vggnet.layers(1:33);

net.layers{1}.filters = 0.01*randn(3,3,1,64, 'single');
net.layers{1}.biases = zeros(1,64,'single');

net.layers{end-1}.filters = 0.01*randn(4,4,512,1024, 'single');
net.layers{end-1}.biases = zeros(1,1024,'single');

end

