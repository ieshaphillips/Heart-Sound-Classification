% 2D Image Processing Spectrogram CNN 

specDir  = fullfile('results','timefreqdomain');   
trainCSV = 'train.csv'; 
valCSV   = 'val.csv'; 
testCSV  = 'test.csv';

Train = readtable(trainCSV,'TextType','string','VariableNamingRule','preserve');
Val   = readtable(valCSV,  'TextType','string','VariableNamingRule','preserve');
Test  = readtable(testCSV, 'TextType','string','VariableNamingRule','preserve');

PTrain = string(Train.PatientID);
PVal   = string(Val.PatientID);
PTest  = string(Test.PatientID);

% spectrogram files 
S = dir(fullfile(specDir,'*_spec.png'));

files  = strings(numel(S),1); 
pids   = strings(numel(S),1);
valves = strings(numel(S),1);      

for k = 1:numel(S)
    files(k) = string(fullfile(S(k).folder, S(k).name));

  
    base  = erase(string(S(k).name), "_spec.png");  
    parts = split(base, "_");                       

    pids(k)   = parts(1);        
    valves(k) = parts(2);        
end

% Match images to Train/Val/Test patients
trP = unique(PTrain); 
vaP = unique(PVal); 
teP = unique(PTest);

isTr = ismember(pids, trP); 
isVa = ismember(pids, vaP); 
isTe = ismember(pids, teP);

% Build image tables (with Valve)
ImgTrain = table(files(isTr), pids(isTr), valves(isTr), ...
    'VariableNames', {'ImgPath','PatientID','Valve'});
ImgVal   = table(files(isVa), pids(isVa), valves(isVa), ...
    'VariableNames', {'ImgPath','PatientID','Valve'});
ImgTest  = table(files(isTe), pids(isTe), valves(isTe), ...
    'VariableNames', {'ImgPath','PatientID','Valve'});

% Build patient -> label map (majority vote over rows)
patLab = containers.Map('KeyType','char','ValueType','double');

% Train 
[G, pU] = findgroups(PTrain);
labs = splitapply(@(z) mode(double(z=="Present")), Train.Murmur, G);
for i = 1:numel(pU)
    key = char(pU(i));   % pU is string array
    patLab(key) = labs(i);
end

%  Val 
[G, pU] = findgroups(PVal);
labs = splitapply(@(z) mode(double(z=="Present")), Val.Murmur, G);
for i = 1:numel(pU)
    key = char(pU(i));
    if ~isKey(patLab, key)
        patLab(key) = labs(i);
    end
end

%  Test 
[G, pU] = findgroups(PTest);
labs = splitapply(@(z) mode(double(z=="Present")), Test.Murmur, G);
for i = 1:numel(pU)
    key = char(pU(i));
    if ~isKey(patLab, key)
        patLab(key) = labs(i);
    end
end

attachLab = @(Tbl) addvars(Tbl, ...
    categorical( arrayfun(@(pid) patLab(char(pid)), Tbl.PatientID), ...
                 [0 1], {'NoMurmur','Murmur'} ), ...
    'NewVariableNames','Label');

ImgTrain = attachLab(ImgTrain);
ImgVal   = attachLab(ImgVal);
ImgTest  = attachLab(ImgTest);

% blancing training set
isMur = (ImgTrain.Label == 'Murmur');
isNo  = (ImgTrain.Label == 'NoMurmur');

nMur = sum(isMur);
nNo  = sum(isNo);

fprintf('Before balancing: Murmur=%d, NoMurmur=%d\n', nMur, nNo);

idxMur = find(isMur);
idxNo  = find(isNo);

% target number of NoMurmur samples
targetNo = nMur;   


% cannot sample more NoMurmur than we actually have
targetNo = min(targetNo, nNo);

rng(1);  % reproducible
if targetNo < nNo
    idxNoSel = idxNo(randperm(nNo, targetNo));  % randomly pick targetNo NoMurmur
else
    idxNoSel = idxNo;                           % use all NoMurmur
end

% keep all murmurs and selected NoMurmur
selIdx = [idxMur; idxNoSel];
ImgTrainBal = ImgTrain(selIdx,:);

% Shuffle rows of balanced training
ImgTrainBal = ImgTrainBal(randperm(height(ImgTrainBal)),:);

nMurBal = sum(ImgTrainBal.Label=='Murmur');
nNoBal  = sum(ImgTrainBal.Label=='NoMurmur');
fprintf('After balancing : Murmur=%d, NoMurmur=%d\n', nMurBal, nNoBal);


imdsTrain = imageDatastore(ImgTrainBal.ImgPath, 'Labels', ImgTrainBal.Label);
imdsVal   = imageDatastore(ImgVal.ImgPath,      'Labels', ImgVal.Label);
imdsTest  = imageDatastore(ImgTest.ImgPath,     'Labels', ImgTest.Label);

inputSize = [224 224 3];  
augTrain  = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'ColorPreprocessing','gray2rgb');
augVal    = augmentedImageDatastore(inputSize(1:2), imdsVal,   'ColorPreprocessing','gray2rgb');
augTest   = augmentedImageDatastore(inputSize(1:2), imdsTest,  'ColorPreprocessing','gray2rgb');

trainCats = categories(removecats(imdsTrain.Labels));
imdsTrain.Labels = reordercats(imdsTrain.Labels, trainCats);
imdsVal.Labels   = reordercats(imdsVal.Labels,   trainCats);
imdsTest.Labels  = reordercats(imdsTest.Labels,  trainCats);
classesCat = categorical(trainCats, trainCats);

% CNN Model 
layers = [
    imageInputLayer(inputSize,"Name","in","Normalization","zscore")

    convolution2dLayer(3,32,"Padding","same","Name","conv1")
    batchNormalizationLayer("Name","bn1")
    reluLayer("Name","relu1")
    maxPooling2dLayer(2,"Stride",2,"Name","pool1")

    convolution2dLayer(3,64,"Padding","same","Name","conv2")
    batchNormalizationLayer("Name","bn2")
    reluLayer("Name","relu2")
    maxPooling2dLayer(2,"Stride",2,"Name","pool2")
    dropoutLayer(0.25,"Name","drop2")

    convolution2dLayer(3,128,"Padding","same","Name","conv3")
    batchNormalizationLayer("Name","bn3")
    reluLayer("Name","relu3")
    maxPooling2dLayer(2,"Stride",2,"Name","pool3")
    dropoutLayer(0.35,"Name","drop3")

    globalAveragePooling2dLayer("Name","gap")
    fullyConnectedLayer(64,"Name","fc1")
    reluLayer("Name","relu4")
    dropoutLayer(0.2,"Name","drop4")

    fullyConnectedLayer(numel(trainCats),"Name","fc_out")
    softmaxLayer("Name","sm")
    classificationLayer("Name","cls","Classes",classesCat)
];

opts = trainingOptions("adam", ...
    "InitialLearnRate",1e-3, ...
    "MiniBatchSize",32, ...
    "MaxEpochs",40, ...
    "Shuffle","every-epoch", ...
    "ValidationData",augVal, ...
    "Verbose",false, ...
    "Plots","training-progress", ...
    "L2Regularization",1e-4);    % try 0, 1e-4, 1e-3, etc.

net = trainNetwork(augTrain, layers, opts);

%%Evaluation
[Yhat, Yprob] = classify(net, augTest);
Ytrue = imdsTest.Labels;

figure;
confusionchart(Ytrue, Yhat, 'Normalization','row-normalized', ...
    'Title','Spectrogram-CNN — Test');

tp = sum(Ytrue=='Murmur'   & Yhat=='Murmur');
tn = sum(Ytrue=='NoMurmur' & Yhat=='NoMurmur');
fp = sum(Ytrue=='NoMurmur' & Yhat=='Murmur');
fn = sum(Ytrue=='Murmur'   & Yhat=='NoMurmur');

sens = tp/(tp+fn); 
spec = tn/(tn+fp);
fprintf('TEST (all valves): Sens=%.3f  Spec=%.3f\n', sens, spec);


% Per Valve Analysis
testValves = ImgTest.Valve;           
valveCats  = unique(testValves);         
fprintf('\nPer-valve performance:\n');
for v = 1:numel(valveCats)
    thisV = valveCats(v);
    mask  = (testValves == thisV);

    Yt_v  = Ytrue(mask);
    Yh_v  = Yhat(mask);

    if isempty(Yt_v)
        continue;
    end

    tp_v = sum(Yt_v=='Murmur'   & Yh_v=='Murmur');
    tn_v = sum(Yt_v=='NoMurmur' & Yh_v=='NoMurmur');
    fp_v = sum(Yt_v=='NoMurmur' & Yh_v=='Murmur');
    fn_v = sum(Yt_v=='Murmur'   & Yh_v=='NoMurmur');

    sens_v = tp_v / max(1,(tp_v+fn_v));
    spec_v = tn_v / max(1,(tn_v+fp_v));

    fprintf('  Valve %s (N=%d): Sens=%.3f  Spec=%.3f\n', ...
        thisV, numel(Yt_v), sens_v, spec_v);
end
