% look into the simulated B-format mixtures for the pattern vs location
clear all; close all; clc
% load the dry signals
[s1,Fs] = audioread('FemaleExample.wav');
[s2,Fs] = audioread('MaleExample.wav');
% load the Bformat RIRs
load('B_format_RIRs_12BB01_Alfredo_S3A_16k.mat');


NFFT = 1024;
Window = 1024;
Shift = 0.25;

L = 16000*4; % 3 second long to have enough data for statistics analysis
I = 2;

sample = 1;
while sample<100,
    % find the I locations with 40 degree constratints for data simulation
    aziarray = [randi(36)];
    for i = 2:I,
        redo = true;
        while(redo)
            azi = randi(36);
            redo = false;
            for j = 1:length(aziarray),
                tempazi = aziarray(j);
                diff = angle(exp(1j*deg2rad((azi-tempazi)*10)));
                if abs(diff)<deg2rad(40)
                    redo = true;
                end
            end
        end
        aziarray = [aziarray,azi];
    end
    
    % generate the mixture
    rir = squeeze(rirs(aziarray(1),:,:))';
    Len = length(s1);
    starti = randi(Len-L); endi = starti+L;
    sig = s1(starti:endi);
    % apply convolution
    p0 = fftfilt(rir(:,1),sig);
    vel_x = fftfilt(rir(:,2),sig);
    vel_y = fftfilt(rir(:,3),sig);
    
    rir = squeeze(rirs(aziarray(2),:,:))';
    Len = length(s2);
    starti = randi(Len-L); endi = starti+L;
    scale = rand(1)*1.5+0.5; %[0.5,3]
    sig = s2(starti:endi)*scale;
    % apply convolution
    p02 = fftfilt(rir(:,1),sig);
    p0 = p0+p02;
    vel_x = vel_x+fftfilt(rir(:,2),sig);
    vel_y = vel_y+fftfilt(rir(:,3),sig);
    
    if I==3,
        rir = squeeze(rirs(aziarray(3),:,:))';
        Len = length(s2);
        starti = randi(Len-L); endi = starti+L;
        scale = rand(1)*1.5+0.5; %[0.5,3]
        sig = s2(starti:endi)*scale;
        %apply convolution
        p03 = fftfilt(rir(:,1),sig);
        p0 = p0+p03;
        vel_x = vel_x+fftfilt(rir(:,2),sig);
        vel_y = vel_y+fftfilt(rir(:,3),sig);
    end
    
    % calculate the gradient angle
    P0 = stft(p0, NFFT, Window, Shift);
    Gx = stft(vel_x, NFFT, Window, Shift);
    Gy = stft(vel_y, NFFT, Window, Shift);
    Y = real(conj(P0).*Gy);
    X = real(conj(P0).*Gx);
    theta = atan2(Y,X);
    
    pspec = abs(P0(:));pspec(abs(P0(:))>0.05) = 0.05;
    % figure;hist(pspec,100);
    threshold = 0.5; % do not use the bottom 10 percent;
    pspecsort = sort(pspec(:),'ascend');
    thresholdval = pspecsort(round(0.1*length(theta)));
    thetause = theta(pspec>thresholdval);
    % now find the peaks
    
    
    [aaa,bbb] = hist(thetause(:),100);
    figure(111);
    plot(bbb,aaa);
    hold on;
    
    aaanew = aaa;
    result = [];
    twosigma2 = 2*0.8^2/4;
    for i = 1:I,
        [~,maxInd] = max(aaanew);
        temp_mean_init = bbb(maxInd);
        
        
        temp_mean = temp_mean_init;
        width = 0.3;
        for it = 1:3,
            diff = thetause-temp_mean;
            diff = angle(exp(1j*diff));
            diff(abs(diff)>width) = [];
            temp_mean = temp_mean+mean(diff);
            temp_mean = angle(exp(1j*temp_mean));
            width = width*0.8;
        end
        result = [result,temp_mean];
        
        % update aaanew
        temp = abs(angle(exp(1j*(bbb-temp_mean))));
        [~,peakInd] = min(temp);
        peakV = aaa(peakInd);
        aaanew = aaanew-peakV*exp(-temp.^2/twosigma2);
        
        %         plot(bbb,aaanew);
        
    end
    
    ylimv = get(gca,'ylim')
    for i = 1:I,
        plot([result(i), result(i)],ylimv);
    end
    
    energy = [norm(p0,'fro'),norm(p02,'fro')];
    if I==3,
        energy = [energy,norm(p03,'fro')];
    end
    
    SNR = 20*log10(max(energy)/min(energy));
    str_e = sprintf('SNR=%0.2f dB',SNR);
    text(-3,ylimv(2)/2,str_e);
    hold off
    
    sample = sample+1;
    pause;
end


% save('Result.mat','Result')
figure;pcolor(Result);shading flat;caxis([-pi pi]);colorbar;colormap([colormap(jet);flipud(colormap(jet))])
