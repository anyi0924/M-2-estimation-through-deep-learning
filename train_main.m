function gpu_depth_local()

mySetup();
gpuDevice(1)
%change output path here
outputDir = 'output';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 4 example settings
trainOpts(1).id = 1;
trainOpts(1).batchSize = 64 ;
trainOpts(1).numEpochs = 30;
trainOpts(1).useGpu = true ;   
% trainOpts(1).dataDir = '../data' ;
trainOpts(1).dataDir = 'data'
trainOpts(1).saveDir = 'save/pattern_vgg' ;
trainOpts(1).outputDir = outputDir ;
depth_train_local(trainOpts(1));

