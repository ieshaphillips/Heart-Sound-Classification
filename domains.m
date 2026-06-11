plotDir = fullfile('all_wav');                
timeDir = fullfile('results','timedomain');   
freqDir = fullfile('results','freqdomain');     
timefreqDir = fullfile('results','timefreqdomain'); 
if ~exist(timeDir,'dir'), mkdir(timeDir); end
if ~exist(freqDir,'dir'), mkdir(freqDir); end
if ~exist(timefreqDir,'dir'), mkdir(timefreqDir); end

wavList = dir(fullfile(plotDir,'*.wav'));
if isempty(wavList), error('No WAV files found in %s', plotDir); end

for k = 1:numel(wavList)
    fpath = fullfile(wavList(k).folder, wavList(k).name);
    [x, fs] = audioread(fpath);

    Win =[20 800] / (fs/2); 
    [b, a] = butter(6, Win); 
    y = filter(b, a, x);

     
    % Time Domain
    t = (0:numel(y)-1)/fs;
    figure1 = figure('Visible','off');
    plot(t, y);
    xlabel('Time (s)'); ylabel('Amplitude');
    title(wavList(k).name, 'Interpreter','none');
    [~, base] = fileparts(wavList(k).name);
    exportgraphics(figure1, fullfile(timeDir, base+"_time.png"), 'Resolution',150);
    close(figure1);
 

    % Frequency Domain
    nfftP = 4096;                
    [Pxx,F] = periodogram(y, [], nfftP, fs);
    PdB = 10*log10(Pxx + 1e-20); 

    figure2 = figure('Visible','off');
    plot(F, PdB); 
    xlim([0 800]);
    xlabel('Frequency (Hz)'); ylabel('Power/Freq (dB/Hz)');
    title(wavList(k).name, 'Interpreter','none');
    exportgraphics(figure2, fullfile(freqDir, base+"_periodogram.png"), 'Resolution',150);
    close(figure2);



    %TimeFreq Domain 
    win  = round(0.06*fs);
    hop  = round(0.01*fs);
    nfft = 1024;
    [S,F,Tstft] = spectrogram(y, hann(win), win-hop, nfft, fs, 'yaxis');
    SdB = 20*log10(abs(S)+1e-12);

    figure3 = figure('Visible','off');
    imagesc(Tstft, F, SdB); axis xy;
    ylim([0 800]);
    xlabel('Time (s)'); ylabel('Hz');
    title(wavList(k).name, 'Interpreter','none');
    caxis([max(SdB(:))-60, max(SdB(:))]);
    exportgraphics(figure3, fullfile(timefreqDir, base+"_spec.png"), 'Resolution',150);
    close(figure3);
end


