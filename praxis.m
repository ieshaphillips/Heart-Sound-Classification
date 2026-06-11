
specDir  = fullfile('results','timefreqdomain');  
trainCSV = 'train.csv'; 
valCSV   = 'val.csv'; 
testCSV  = 'test.csv';

% Read splits 
TrainT = readtable(trainCSV,'TextType','string','VariableNamingRule','preserve');
ValT   = readtable(valCSV,  'TextType','string','VariableNamingRule','preserve');
TestT  = readtable(testCSV, 'TextType','string','VariableNamingRule','preserve');

yTrain_pat = double(TrainT.Murmur=="Present");
yVal_pat   = double(ValT.Murmur  =="Present");
yTest_pat  = double(TestT.Murmur =="Present");

%% ------------------ Metadata encoding -----------------------
% Normalize column names you need
TrainT.PregnancyStatus = string(TrainT.("Pregnancy status"));
ValT.PregnancyStatus   = string(ValT.("Pregnancy status"));
TestT.PregnancyStatus  = string(TestT.("Pregnancy status"));

numCols = {'Height','Weight'};
catCols = {'Sex','Age','PregnancyStatus'};

% Fit z-score (TRAIN only)
mu = struct(); sd = struct();
for i=1:numel(numCols)
    c = numCols{i}; v = double(TrainT.(c));
    mu.(c) = mean(v,'omitnan'); s = std(v,0,'omitnan'); if s==0, s=1; end
    sd.(c) = s;
end
% Fit category sets (TRAIN only)
enc = struct();
for i=1:numel(catCols)
    c = catCols{i};
    enc.(c) = categories(categorical(TrainT.(c)));
end

% Row encoder (numeric z-score + one-hot with fixed cats)
encRow = @(row) [ ...
    (double(row.Height)-mu.Height)/sd.Height, ...
    (double(row.Weight)-mu.Weight)/sd.Weight, ...
    localOneHot(string(row.Sex), enc.Sex), ...
    localOneHot(string(row.Age), enc.Age), ...
    localOneHot(string(row.PregnancyStatus), enc.PregnancyStatus) ...
    ];
% Build patient -> metadata vector map (first occurrence per patient)
pat2meta = containers.Map('KeyType','char','ValueType','any');

% Helper to add rows from a table if the Patient ID is not already in the map
function addPatientsToMap(Tin, pat2meta, encRow)
    for i = 1:height(Tin)
        pid = char(string(Tin.("Patient ID")(i)));
        if ~isKey(pat2meta, pid)
            pat2meta(pid) = encRow(Tin(i,:));   % <-- assignment, not '=='
        end
    end
end

% Populate from Train first (defines categories/stats), then Val/Test
addPatientsToMap(TrainT, pat2meta, encRow);
addPatientsToMap(ValT,   pat2meta, encRow);
addPatientsToMap(TestT,  pat2meta, encRow);

% Number of metadata features (use any key that exists)
someKey = pat2meta.keys;
numMeta = numel(pat2meta(someKey{1}));

%% ------------------ Spectrogram files index -----------------
D = dir(fullfile(specDir,'*_spec.png'));
if isempty(D), error('No *_spec.png found in %s', specDir); end

files = strings(numel(D),1); pids = strings(numel(D),1);
for k=1:numel(D)
    files(k) = string(fullfile(D(k).folder, D(k).name));
    base = erase(string(D(k).name), "_spec.png");
    pids(k) = extractBefore(base, "_");     % PatientID before first underscore
end

% Split membership by patient
trP = string(unique(TrainT.("Patient ID")));
vaP = string(unique(ValT.("Patient ID")));
teP = string(unique(TestT.("Patient ID")));

isTr = ismember(pids, trP); 
isVa = ismember(pids, vaP); 
isTe = ismember(pids, teP);

ImgTrain = table(files(isTr), pids(isTr), 'VariableNames', {'ImgPath','PatientID'});
ImgVal   = table(files(isVa), pids(isVa), 'VariableNames', {'ImgPath','PatientID'});
ImgTest  = table(files(isTe), pids(isTe), 'VariableNames', {'ImgPath','PatientID'});

% Patient-level labels (majority over rows in split tables)
patLab = containers.Map('KeyType','char','ValueType','double');
% Train
[G, pU] = findgroups(string(TrainT.("Patient ID")));
labs = splitapply(@(z) mode(double(z=="Present")), TrainT.Murmur, G);
for i=1:numel(pU), patLab(char(pU(i))) = labs(i); end
% Val
[G, pU] = findgroups(string(ValT.("Patient ID")));
labs = splitapply(@(z) mode(double(z=="Present")), ValT.Murmur, G);
for i=1:numel(pU), if ~isKey(patLab,char(pU(i))), patLab(char(pU(i))) = labs(i); end, end
% Test
[G, pU] = findgroups(string(TestT.("Patient ID")));
labs = splitapply(@(z) mode(double(z=="Present")), TestT.Murmur, G);
for i=1:numel(pU), if ~isKey(patLab,char(pU(i))), patLab(char(pU(i))) = labs(i); end, end

% Attach labels + metadata rows per image
attach = @(Tbl) addvars(Tbl, ...
    arrayfun(@(pid) patLab(char(pid)), Tbl.PatientID), ...
    'NewVariableNames','y');
ImgTrain = attach(ImgTrain); ImgVal = attach(ImgVal); ImgTest = attach(ImgTest);

metaRows = @(Tbl) cell2mat( arrayfun(@(pid) pat2meta(char(pid)), Tbl.PatientID, 'UniformOutput',false) );
Mtr = metaRows(ImgTrain); Mva = metaRows(ImgVal); Mte = metaRows(ImgTest);

%% ------------------ Spectrogram feature extraction (fixed) ----------
% Get length from the first TRAIN file
assert(~isempty(ImgTrain), 'ImgTrain is empty.');
f0 = specfeat_from_png(ImgTrain.ImgPath{1});
Dspec = numel(f0);

% --- TRAIN ---
Xtr_spec = zeros(height(ImgTrain), Dspec, 'single');
Xtr_spec(1,:) = f0;
for i = 2:height(ImgTrain)
    f = specfeat_from_png(ImgTrain.ImgPath{i});
    if numel(f) ~= Dspec
        f = fixlen(f, Dspec);               % pad/truncate if needed
    end
    Xtr_spec(i,:) = f;
end

% --- VAL ---
Xva_spec = zeros(height(ImgVal), Dspec, 'single');
for i = 1:height(ImgVal)
    f = specfeat_from_png(ImgVal.ImgPath{i});
    if numel(f) ~= Dspec
        f = fixlen(f, Dspec);
    end
    Xva_spec(i,:) = f;
end

% --- TEST ---
Xte_spec = zeros(height(ImgTest), Dspec, 'single');
for i = 1:height(ImgTest)
    f = specfeat_from_png(ImgTest.ImgPath{i});
    if numel(f) ~= Dspec
        f = fixlen(f, Dspec);
    end
    Xte_spec(i,:) = f;
end


% Optional PCA (fit on TRAIN only; project val/test)
if ~isempty(Xtr_spec)
    [coeff,score,~,~,expl,muSpec] = pca(double(Xtr_spec));
    k = find(cumsum(expl)>=95,1); if isempty(k), k = size(score,2); end
    Xtr_spec = single(score(:,1:k));
    Xva_spec = single( (double(Xva_spec)-muSpec) * coeff(:,1:k) );
    Xte_spec = single( (double(Xte_spec)-muSpec) * coeff(:,1:k) );
end

%% ------------------ Assemble final feature matrices ----------
Xtr = [single(Mtr), single(Xtr_spec)];
Xva = [single(Mva), single(Xva_spec)];
Xte = [single(Mte), single(Xte_spec)];

ytr = double(ImgTrain.y(:));    % ensure double 0/1
yva = double(ImgVal.y(:));
yte = double(ImgTest.y(:));

fprintf('Feature dims: meta=%d, spec=%d -> total=%d\n', size(Mtr,2), size(Xtr_spec,2), size(Xtr,2));

%% ------------------ CLEAN & STANDARDIZE FEATURES -------------
[Xtr, muX, sdX] = cleanAndZ(Xtr);
Xva = applyZ(Xva, muX, sdX);
Xte = applyZ(Xte, muX, sdX);

% Class weighting (handle single-class edge case)
nPos = sum(ytr==1); nNeg = sum(ytr==0);
if nPos==0 || nNeg==0
    warning('Train set single-class (pos=%d, neg=%d). Using posW=1.', nPos, nNeg);
    posW = 1;
else
    posW = nNeg / max(1,nPos);          % ratio
end

%% ------------------ Custom logistic: BCE + L2 + Adam --------
[Ntr,D] = size(Xtr);
W = 0.01*randn(D,1,'single'); 
b = single(0);

lr = 5e-4;                    % slightly lower LR helps stability
epochs = 120;                 % a few more epochs
mb = 64; 
lambda = 1e-5;

mW = zeros(D,1,'single'); vW = zeros(D,1,'single');
mbias = single(0); vbias = single(0);
beta1=0.9; beta2=0.999; eps=1e-8;
sig = @(z) 1./(1+exp(-z));

idxAll = (1:Ntr)';

for ep=1:epochs
    idx = idxAll(randperm(Ntr));
    for s=1:mb:Ntr
        j = idx(s:min(s+mb-1,Ntr));
        Xb = Xtr(j,:); yb = single(ytr(j));
        B  = size(Xb,1);

        % -------- forward (stable) --------
        z = Xb*W + b; 
        z(~isfinite(z)) = 0;
        z = max(min(z,20),-20);       % logit clip
        p = sig(z);
        p = min(max(p,1e-7),1-1e-7);  % prob clamp

        % -------- weighted, stable BCE --------
        % BCE(y,z) = max(0,z) - z*y + log1p(exp(-abs(z)))
        bce = max(0,z) - z.*yb + log1p(exp(-abs(z)));
        w = ones(B,1,'single'); w(yb==1) = posW;

        dataLoss = sum(w.*bce)/B;
        regLoss  = 0.5*lambda*sum(W.^2);
        loss = dataLoss + regLoss;

        % -------- gradients --------
        dz = w.*(p - yb)/B;                 % same as logistic derivative
        gW = Xb.'*dz + lambda*W;
        gb = sum(dz);

        % -------- Adam updates --------
        mW = beta1*mW + (1-beta1)*gW;       vW = beta2*vW + (1-beta2)*(gW.^2);
        mWh = mW/(1-beta1^ep);              vWh = vW/(1-beta2^ep);
        W = W - lr*mWh./(sqrt(vWh)+eps);

        mbias = beta1*mbias + (1-beta1)*gb; vbias = beta2*vbias + (1-beta2)*(gb.^2);
        mbh = mbias/(1-beta1^ep);           vbh = vbias/(1-beta2^ep);
        b = b - lr*mbh/(sqrt(vbh)+eps);
    end

    if mod(ep,10)==0
        pv = sig(max(min(Xva*W+b,20),-20));
        acc = mean((pv>=0.5)==logical(yva));
        fprintf('Epoch %3d | val acc=%.3f | loss≈%.5f\n', ep, acc, double(loss));
    end
end

%% ------------------ Threshold tuning on VAL ------------------
pv = sig(max(min(Xva*W+b,20),-20));
ths = linspace(0.05,0.95,19);
bestF1=0; bestTh=0.5; yt = logical(yva);
for th = ths
    yh = pv>=th; tp=sum(yt&yh); fp=sum(~yt&yh); fn=sum(yt&~yh);
    f1 = 2*tp / max(1,2*tp+fp+fn);
    if f1>bestF1, bestF1=f1; bestTh=th; end
end
fprintf('Chosen threshold (val): %.2f (F1=%.3f)\n', bestTh, bestF1);

%% ------------------ Test metrics + Confusion Matrix ----------
pt = sig(max(min(Xte*W+b,20),-20));
yh = pt>=bestTh; 
yt = logical(yte);

% confusion counts
tn = sum(~yt & ~yh);
fp = sum(~yt &  yh);
fn = sum( yt & ~yh);
tp = sum( yt &  yh);

sens = tp/(tp+fn);
spec = tn/(tn+fp);
if numel(unique(yte)) < 2
    auroc = NaN;
else
    [~,~,~,auroc] = perfcurve(yte, double(pt), 1);
end
fprintf('TEST: Sens=%.3f  Spec=%.3f  AUROC=%.3f  (thr=%.2f)\n', sens,spec,auroc,bestTh);

% Pretty confusion matrix figure (saved)
YtrueCat = categorical(yte,[0 1],{'NoMurmur','Murmur'});
YhatCat  = categorical(yh, [0 1],{'NoMurmur','Murmur'});

figure('Color','w');
confusionchart(YtrueCat, YhatCat, ...
    'Normalization','row-normalized', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized', ...
    'Title','Confusion Matrix – Test (Logistic+Meta+Spec)');
if ~exist('results','dir'), mkdir('results'); end
exportgraphics(gcf,'results/confusion_matrix_test_logistic.png','Resolution',200);

% Also print raw matrix + derived metrics
[C, order] = confusionmat(YtrueCat, YhatCat);
disp(array2table(C,'VariableNames',"Pred_"+order,'RowNames',"True_"+order));
prec = tp/(tp+fp+eps);
f1   = 2*tp/(2*tp+fp+fn+eps);
fprintf('TEST: Prec=%.3f  F1=%.3f\n', prec, f1);

%% ------------------ Helpers (place once in your file) --------
function [X,mu,sd] = cleanAndZ(X)
    X(~isfinite(X)) = NaN;                 % Inf -> NaN
    % median impute per column
    for j=1:size(X,2)
        col = X(:,j);
        m = median(col,'omitnan'); if ~isfinite(m), m=0; end
        col(~isfinite(col)) = m;
        X(:,j) = col;
    end
    mu = mean(X,1); sd = std(X,0,1); sd(sd==0)=1;
    X = (X - mu)./sd;
    X(~isfinite(X)) = 0;
    X = single(X);
end

function X = applyZ(X,mu,sd)
    X(~isfinite(X)) = NaN;
    for j=1:size(X,2)
        col = X(:,j);
        m = median(col,'omitnan'); if ~isfinite(m), m=0; end
        col(~isfinite(col)) = m;
        X(:,j) = col;
    end
    sd(sd==0)=1;
    X = (X - mu)./sd;
    X(~isfinite(X)) = 0;
    X = single(X);
end

%% ===================== HELPERS ==============================
function one = localOneHot(val, cats)
% one-hot with fixed category set (unseen -> all zeros)
one = zeros(1,numel(cats),'single');
idx = find(strcmp(cats, val), 1);
if ~isempty(idx), one(idx)=1; end
end

function f = specfeat_from_png(p)
% Return 1×(F*5) vector: per-frequency {mean,std,median,p90,max} over time
I = imread(p);
if ndims(I)==3, I = rgb2gray(I); end
A = im2single(I);           % [F × T] (freq × time)
% Normalize per-image for stability
A = (A - min(A(:))) / max(1e-6, (max(A(:))-min(A(:))));
m  = mean(A,2,'omitnan');         % F×1
s  = std(A,0,2,'omitnan');
md = median(A,2,'omitnan');
p90= prctile(A,90,2);
mx = max(A,[],2);
f = single([m; s; md; p90; mx]).';
end
function y = fixlen(x, D)
% Ensure row vector length D (pad with zeros or truncate)
x = x(:).';
if numel(x) == D
    y = x;
elseif numel(x) > D
    y = x(1:D);
else
    y = [x, zeros(1, D - numel(x), 'like', x)];
end
end
