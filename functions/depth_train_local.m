function depth_train_local(trainOpts)

saveid = num2str(trainOpts.id, '%02d');
trainOpts.outputDir = [trainOpts.outputDir '/pattern_vgg_' saveid ];
if ~exist(trainOpts.outputDir, 'dir'), mkdir(trainOpts.outputDir) ; end

% trainOpts.saveDir = [trainOpts.saveDir '/trainDAG_' saveid];
if ~exist(trainOpts.saveDir, 'dir'), mkdir(trainOpts.saveDir) ; end

%Loading the training data

% Take the average image out
% imdb = load([trainOpts.dataDir '/' 'nyudb.mat']) ;

% for testing
myTrain(trainOpts);
% net.imageMean = imageMean;

% save([netSavepath '/' 'train-net.mat'], 'net');
% save([netSavepath '/' 'info.mat'], 'info');
% save([netSavepath '/' 'trainOpts.mat'], 'trainOpts');

end

