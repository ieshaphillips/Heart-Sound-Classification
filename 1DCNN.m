%% 1D Time Domain CNN 

pcgRootDir = fullfile('all_wav');   
pcgExt     = '.wav';               

% Columns in CSV 
pidCol = "PatientID";          
locCol = "RecordingLocations_"; % valve locations column 

%sample rate and length per recording
fsTarget = 2000;  % Hz
L        = 4000;  % number of time samples (e.g., 2 s @ 2 kHz)


trainCSV = 'train.csv';
valCSV   = 'val.csv';
testCSV  = 'test.csv';

Train = readtable(trainCSV,'TextType','string','VariableNamingRule','preserve');
Val   = readtable(valCSV,  'TextType','string','VariableNamingRule','preserve');
Test  = readtable(testCSV, 'TextType','string','VariableNamingRule','preserve');

% Labels from Murmur column: 0 = NoMurmur, 1 = Murmur
yTrain = double(Train.Murmur=="Present");
yVal   = double(Val.Murmur  =="Present");
yTest  = double(Test.Murmur =="Present");

YTrainCat = categorical(yTrain,[0 1],{'NoMurmur','Murmur'});
YValCat   = categorical(yVal,  [0 1],{'NoMurmur','Murmur'});
YTestCat  = categorical(yTest, [0 1],{'NoMurmur','Murmur'});

fprintf('TRAIN rows (original): Murmur=%d, NoMurmur=%d\n', ...
    sum(yTrain==1), sum(yTrain==0));


fprintf('Building PCG sequences for TRAIN...\n');
XTrainSeq = buildPCGSeqSet(Train, pcgRootDir, pcgExt, pidCol, locCol, fsTarget, L);

fprintf('Building PCG sequences for VAL...\n');
XValSeq   = buildPCGSeqSet(Val,   pcgRootDir, pcgExt, pidCol, locCol, fsTarget, L);

fprintf('Building PCG sequences for TEST...\n');
XTestSeq  = buildPCGSeqSet(Test,  pcgRootDir, pcgExt, pidCol, locCol, fsTarget, L);


% blancing training set

idxMur = find(yTrain==1);   % murmurs
idxNo  = find(yTrain==0);   % no murmurs

nMur = numel(idxMur);
nNo  = numel(idxNo);

fprintf('Before balancing: Murmur=%d, NoMurmur=%d\n', nMur, nNo);

targetNo = nMur - 25;   

targetNo = min(targetNo, nNo);

if targetNo < nNo
    idxNoKeep = idxNo(randperm(nNo, targetNo));  % randomly pick targetNo NoMurmur
else
    idxNoKeep = idxNo;                            % use all NoMurmur
end

% keeps al murmurs and selected no murmur
keepIdx = sort([idxMur; idxNoKeep]);

XTrainSeqBal = XTrainSeq(keepIdx);
YTrainCatBal = YTrainCat(keepIdx);

nMurBal = sum(YTrainCatBal=='Murmur');
nNoBal  = sum(YTrainCatBal=='NoMurmur');
fprintf('After balancing: Murmur=%d, NoMurmur=%d\n', nMurBal, nNoBal);


% CNN Model 
numChannels = 1; %1

layers = [
    sequenceInputLayer(numChannels, ...
        "Name","seq_in", ...
        "MinLength",L)   % ensure pooling is valid

    convolution1dLayer(7,64,"Padding","same","Name","conv1")
    batchNormalizationLayer("Name","bn1")
    reluLayer("Name","relu1")
    maxPooling1dLayer(2,"Stride",2,"Name","pool1")

    convolution1dLayer(7,128,"Padding","same","Name","conv2")
    batchNormalizationLayer("Name","bn2")
    reluLayer("Name","relu2")
    maxPooling1dLayer(2,"Stride",2,"Name","pool2")
    dropoutLayer(0.3,"Name","drop2")

    convolution1dLayer(5,128,"Padding","same","Name","conv3")
    batchNormalizationLayer("Name","bn3")
    reluLayer("Name","relu3")
    maxPooling1dLayer(2,"Stride",2,"Name","pool3")
    dropoutLayer(0.4,"Name","drop3")

    globalAveragePooling1dLayer("Name","gap")

    fullyConnectedLayer(64,"Name","fc1")
    reluLayer("Name","relu4")
    dropoutLayer(0.3,"Name","drop4")

    fullyConnectedLayer(2,"Name","fc_out")   % 2 classes
    softmaxLayer("Name","sm")
    classificationLayer("Name","cls")
];

miniBatchSize = 64; 

opts = trainingOptions("adam", ...
    "InitialLearnRate",3e-4, ...
    "MiniBatchSize", miniBatchSize, ...
    "MaxEpochs",60, ...
    "Shuffle","every-epoch", ...
    "ValidationData",{XValSeq, YValCat}, ...
    "Verbose",false, ...
    "Plots","training-progress", ...
    "L2Regularization",5e-4, ...
    "ValidationPatience",10);

net1D = trainNetwork(XTrainSeqBal, YTrainCatBal, layers, opts);

% Evaluation 
YhatTest = classify(net1D, XTestSeq);

figure;
confusionchart(YTestCat, YhatTest, ...
    'Normalization','row-normalized', ...
    'Title','1D CNN on PCG Waveforms — Test');

tp = sum(YTestCat=='Murmur'   & YhatTest=='Murmur');
tn = sum(YTestCat=='NoMurmur' & YhatTest=='NoMurmur');
fp = sum(YTestCat=='NoMurmur' & YhatTest=='Murmur');
fn = sum(YTestCat=='Murmur'   & YhatTest=='NoMurmur');

sens = tp/(tp+fn);
spec = tn/(tn+fp);
fprintf('PCG 1D CNN TEST: Sens=%.3f  Spec=%.3f\n', sens, spec);

%Per Valve Analysis 
testValves = strings(height(Test),1);

for i = 1:height(Test)
    locRaw = string(Test.(locCol)(i));
    if contains(locRaw, '+')
        parts = split(locRaw, '+');
        testValves(i) = strtrim(parts(1));
    else
        testValves(i) = strtrim(locRaw);
    end
end

valveCats = unique(testValves);

fprintf('\nPer-valve performance (1D PCG CNN, TEST):\n');
for v = 1:numel(valveCats)
    thisV = valveCats(v);
    mask  = (testValves == thisV);

    Yt_v  = YTestCat(mask);
    Yh_v  = YhatTest(mask);

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




function XSeq = buildPCGSeqSet(T, pcgRootDir, pcgExt, pidCol, locCol, fsTarget, L)
% Build a cell array of PCG sequences from a table T
% Each cell: [numChannels x L] (here numChannels=1)

    n = height(T);
    XSeq = cell(n,1);

    for i = 1:n
        pidRaw = string(T.(pidCol)(i));
        locRaw = string(T.(locCol)(i));

        % Some rows have 'AV+PV+TV+MV' etc. Use ONLY the first valve (e.g. 'AV').
        if contains(locRaw, '+')
            parts = split(locRaw, '+');
            locOne = strtrim(parts(1));
        else
            locOne = strtrim(locRaw);
        end

        baseName = pidRaw + "_" + locOne + pcgExt;
        fPath    = fullfile(pcgRootDir, baseName);

        if ~isfile(fPath)
            warning('Missing PCG file: %s (row %d). Using zeros.', fPath, i);
            seq = zeros(1,L,'single');
        else
            seq = loadPCG1D(fPath, fsTarget, L);
        end

        XSeq{i} = seq;  % [1 x L]
    end
end

function seq = loadPCG1D(fPath, fsTarget, L)
% Load audio file, bandpass 20–800 Hz, resample, pad/trim to length L

    [x, fs] = audioread(fPath);
    if size(x,2) > 1
        x = mean(x,2);          % mono
    end
    x = single(x(:));           % column

    % Band-pass 20–800 Hz
    fLo = 20;
    fHi = 800;
    [b,a] = butter(4, [fLo fHi]/(fs/2), 'bandpass');
    x = filtfilt(b,a,double(x));
    x = single(x);

    % Resample to fsTarget
    if fs ~= fsTarget
        x = resample(x, fsTarget, fs);
    end

    % Normalize per recording
    x = x - mean(x,'omitnan');
    m = max(abs(x));
    if m > 0
        x = x / m;
    end

    % Pad or trim to L
    if numel(x) < L
        x = [x; zeros(L-numel(x),1,'like',x)];
    else
        x = x(1:L);
    end

    % Return as [1 x L] (1 channel)
    seq = reshape(x, [1, L]);
end
