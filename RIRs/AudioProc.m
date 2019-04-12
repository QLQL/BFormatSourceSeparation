% clear all; close all; clc
%
% % concatnate a group of short audio sequences from the TIMIT dataset
% sig = [];
% dir1 = '/DirectoryToTIMIT/TIMIT/Clean/timit/train/dr3/fdfb0';
% dir2 = '/DirectoryToTIMIT/TIMIT/Clean/timit/train/dr3/fntb0';
% name = 'FemaleExample.wav';
%
% % dir1 = '/DirectoryToTIMIT/audio/TIMIT/Clean/timit/train/dr3/mdtb0';
% % dir2 = '/DirectoryToTIMIT/audio/TIMIT/Clean/timit/train/dr3/mjjb0';
% % name = 'MaleExample.wav';
% for dir_i = {dir1,dir2}
%     audio_files=dir(fullfile(dir_i{1},'*.wav'));
%     for k = 1 : length(audio_files)
%         baseFileName = audio_files(k).name;
%         fullFileName = fullfile(audio_files(k).folder, baseFileName);
%
%         [temp,Fs] = audioread(fullFileName);
%         temp = temp/(max(abs(temp))*2);
%         sig = [sig,temp'];
%     end
% end
% % audiowrite(name,sig,Fs);
%
% aa = 0;





% % look into the simulated B-format mixtures for the pattern vs location
% clear all; close all; clc
% % load the dry signals
% [s1,Fs] = audioread('FemaleExample.wav');
% [s2,Fs] = audioread('MaleExample.wav');
% Len = min(length(s1),length(s2));
% s1 = s1(1:Len);
% s2 = s2(1:Len);
% % load the Bformat RIRs
% load('B_format_RIRs_12BB01_Alfredo_S3A_16k.mat');
%
%
% NFFT = 1024;
% Window = 1024;
% Shift = 0.25;
% sig = s1;
% Result = zeros(36,513);
%
% for azi = 0:10:350,%[0,170:10:220],%
%     fprintf('\n The azimuth of the speaker is %d +++++++++++++++++++\n', azi);
%     azi_i = round(azi/10)+1;
%     rir = squeeze(rirs(azi_i,:,:))';
%
%     % apply convolution
%     p0 = fftfilt(rir(:,1),sig);
%     vel_x = fftfilt(rir(:,2),sig);
%     vel_y = fftfilt(rir(:,3),sig);
%
%     % calculate the gradient angle
%     P0 = stft(p0, NFFT, Window, Shift);
%     Gx = stft(vel_x, NFFT, Window, Shift);
%     Gy = stft(vel_y, NFFT, Window, Shift);
%     Y = real(conj(P0).*Gy);
%     X = real(conj(P0).*Gx);
%     theta = atan2(Y,X);
%
%     %     threshold = median(abs(P0(:)));
%     %     threshold =-10000000;
%
%     [aaa,bbb] = hist(theta(:),100);
%     [~,maxInd] = max(aaa);
%     temp_mean_init = bbb(maxInd);
%
%     for f = 1:size(P0,1),
%         temp = P0(f,:);
%         tempang = theta(f,:);
%         tempsort = sort(temp,'ascend');
%         threshold = tempsort(round(0.5*length(tempsort)));
%
%         useData = tempang(temp>threshold);
%         if any(useData)
%             temp_mean = temp_mean_init;
%             for it = 1:3,
%                 diff = useData-temp_mean;
%                 diff(diff>pi) = diff(diff>pi)-2*pi;
%                 diff(diff<-pi) = diff(diff<-pi)+2*pi;
%                 diff(abs(diff)>0.5) = [];
%                 temp_mean = temp_mean+mean(diff);
%                 if temp_mean>pi,temp_mean = temp_mean-2*pi;end
%                 if temp_mean<-pi,temp_mean = temp_mean+2*pi;end
%             end
%             Result(azi_i,f) = temp_mean;
%         end
%     end
%
% %     figure;hist(theta(:),100);
% %     aa = colormap(jet);
% %     figure;pcolor(theta(:,1:200));shading interp;caxis([-pi pi]);colorbar;colormap([aa;flipud(aa)])
%
% end
% % save('Result.mat','Result')
% figure;pcolor(Result);shading flat;caxis([-pi pi]);colorbar;colormap([colormap(jet);flipud(colormap(jet))])


% clear all; close all; clc
% load('ApatialAnglePatternResult.mat')
% figure('position',[500 500 400 300]);
% pcolor(Result);shading flat;caxis([-pi pi]);colormap(hsv);
% % phasemap(64)
% phasebar('size', 0.4, 'location', 'se');
% xticks([1 128 256 384 512]);
% xticklabels({'0','2', '4', '6', '8'});
% xlabel('Frequency [kHz]');
%
% yticks([1 7:6:36 36]);
% yticklabels(cellstr(num2str([0,60:60:350, 360]'))');
% ylabel('Input angle [deg]')
%
% ResultMean = zeros(36,1);
% for azi_i = 1:36,
%     useData = Result(azi_i,:);
%     [aaa,bbb] = hist(useData(:),20);
%     [~,maxInd] = max(aaa);
%     temp_mean_init = bbb(maxInd);
%     fprintf('\n The azimuth of the speaker is %d +++++++++++++++++++, intial angle %f pi \n', (azi_i-1)*10,temp_mean_init/pi);
%
%     temp_mean = temp_mean_init;
%     for it = 1:3,
%         diff = useData-temp_mean;
%         diff(diff>pi) = diff(diff>pi)-2*pi;
%         diff(diff<-pi) = diff(diff<-pi)+2*pi;
%         diff(abs(diff)>0.5) = [];
%         temp_mean = temp_mean+mean(diff);
%
%         if temp_mean>pi,temp_mean = temp_mean-2*pi;end
%         if temp_mean<-pi,temp_mean = temp_mean+2*pi;end
%     end
%     ResultMean(azi_i) = temp_mean;
%
% end
% % save('SpatialFeatureNorm.mat','ResultMean')




% get the sigma2 of theta distribution
% look into the simulated B-format mixtures for the pattern vs location

clear all; close all; clc
load('B_format_RIRs_12BB01_Alfredo_S3A_16k.mat');
[sig,Fs] = audioread('FemaleExample.wav');
azi = 30;
histMean = pi/6-0.2;
NFFT = 1024;
Window = 1024;
Shift = 0.25;
fprintf('\n The azimuth of the speaker is %d +++++++++++++++++++\n', azi);
azi_i = round(azi/10)+1;
rir = squeeze(rirs(azi_i,:,:))';

% apply convolution
p0 = fftfilt(rir(:,1),sig);
vel_x = fftfilt(rir(:,2),sig);
vel_y = fftfilt(rir(:,3),sig);

% calculate the gradient angle
P0 = stft(p0, NFFT, Window, Shift);
Gx = stft(vel_x, NFFT, Window, Shift);
Gy = stft(vel_y, NFFT, Window, Shift);
Y = real(conj(P0).*Gy);
X = real(conj(P0).*Gx);
theta = atan2(Y,X);

load('ApatialAnglePatternResult.mat')
figure('position',[500 500 400 300]);
hh1 = subplot(211);
% thetaall = theta(:);
% useIndex = abs(thetaall-histMean)<pi/3;
% histfit(thetaall(useIndex),100)
hh = histogram(theta(:),100,'Normalization','probability','FaceColor', [.6 .6 .6]);
hold on
xlim([-1.01*pi,1.01*pi]);
xticks([-pi -0.5*pi 0 pi/6-0.02 0.5*pi pi]);
xticklabels({'-\pi','-.5\pi', '0', '\pi/6', '.5\pi', '\pi'});
xlabel('Angular feature \theta [rad]')
ylabel('\theta distribution')
ylim([0 0.08])
plot([pi/6-0.02 pi/6-0.02],[0 0.08],'r','LineWidth',2);
legend(hh,'Input angle: \pi/6', 'location', 'nw');

hh2 = subplot(212);
aa = pcolor(Result);shading flat;caxis([-pi pi]);colormap(hsv);
set(aa,'facealpha',0.8)
% phasemap(64)
phasebar('size', 0.8, 'location', 'se');
xticks([1 128 256 384 512]);
xticklabels({'0','2', '4', '6', '8'});
xlabel('Frequency [kHz]');

% yticks([1 7:6:36 36]);
% yticklabels(cellstr(num2str([0,60:60:350, 360]'))');
yticks([1 10 19 28 36]);
yticklabels({'0', '.5\pi', '\pi', '1.5\pi', '2\pi'});
ylabel('Input angle [rad]');
set(hh1,'position',[0.15,0.7,0.82,0.25]);
set(hh2,'position',[0.15,0.12,0.82,0.45]);



% load the dry signals
[s1,Fs] = audioread('FemaleExample.wav');
[s2,Fs] = audioread('MaleExample.wav');
Len = min(length(s1),length(s2));
s1 = s1(1:Len);
s2 = s2(1:Len);
% load the Bformat RIRs
load('B_format_RIRs_12BB01_Alfredo_S3A_16k.mat');
load('B_format_RIRs_SpatialFeatureNorm.mat')


NFFT = 1024;
Window = 1024;
Shift = 0.25;

aziarray = [0   60 240 30;
    90 180 300 330];
aziarraystr = {'0'  '\pi/3' '-2\pi/3' '\pi/6';
    '\pi/2' '\pi' '-\pi/3' '-\pi/6'};

% n_colors = 6;
% colors = distinguishable_colors(n_colors);
% to be consistent with the phase bar color

allcolors = colormap(hsv);
n_colors = 8;
colors = zeros(n_colors,3);
for ii = 1:4,
    temp = ceil((aziarray(:,ii)+0.1)/360*length(allcolors));
    temp = temp+length(allcolors)/2;
    temp(temp>length(allcolors)) = temp(temp>length(allcolors))-length(allcolors);
    colors(ii*2-1:ii*2,:) = allcolors(temp,:);
end

figure('position',[1000 200 900 600])
HH = [];
for ii = 1:4,
    useColors = colors(ii*2-1:ii*2,:);
    azi1 = aziarray(1,ii);
    azi2 = aziarray(2,ii);
    
    azi_1 = round(azi1/10)+1;
    rir_1 = squeeze(rirs(azi_1,:,:))';
    azi_2 = round(azi2/10)+1;
    rir_2 = squeeze(rirs(azi_2,:,:))';
    
    % apply convolution
    p0_1 = fftfilt(rir_1(:,1),s1);
    vel_x_1 = fftfilt(rir_1(:,2),s1);
    vel_y_1 = fftfilt(rir_1(:,3),s1);
    
    p0_2 = fftfilt(rir_2(:,1),s2);
    vel_x_2 = fftfilt(rir_2(:,2),s2);
    vel_y_2 = fftfilt(rir_2(:,3),s2);
    
    p0 = p0_1+p0_2;
    vel_x = vel_x_1 + vel_x_2;
    vel_y = vel_y_1 + vel_y_2;
    
    % calculate the gradient angle
    P0 = stft(p0, NFFT, Window, Shift);
    Gx = stft(vel_x, NFFT, Window, Shift);
    Gy = stft(vel_y, NFFT, Window, Shift);
    Y = real(conj(P0).*Gy);
    X = real(conj(P0).*Gx);
    theta = atan2(Y,X);
    
    P0_1 = stft(p0_1, NFFT, Window, Shift);
    P0_2 = stft(p0_2, NFFT, Window, Shift);
    mask = abs(P0_1)>abs(P0_2);
    
    % threshold = median(abs(P0(:)));
    threshold = -100000;
    useIndex = abs(P0)>threshold;
    
    
    
    
    u1 = ResultMean(round(azi1/10)+1);
    temp = theta-u1;
    temp(temp>pi) = temp(temp>pi)-pi*2;temp(temp<-pi) = temp(temp<-pi)+pi*2;
    theta_shift1 = exp(-temp.^2/(2*1.5));
    u2 = ResultMean(round(azi2/10)+1);
    temp = theta-u2;
    temp(temp>pi) = temp(temp>pi)-pi*2;temp(temp<-pi) = temp(temp<-pi)+pi*2;
    theta_shift2 = exp(-temp.^2/(2*1.5));
    
    
    
    thetaUse_1 = theta(useIndex & mask);
    thetaUse_2 = theta(useIndex & ~mask);
    
    htemp = subplot(3,4,ii);
    HH = [HH,htemp];
    histogram(theta(:),100,'LineStyle','None','FaceColor',[1,1,1]*0.5);
    hold on
    histogram(thetaUse_1,100,'LineStyle','None','FaceColor',useColors(1,:)*0.5+0.5);
    histogram(thetaUse_2,100,'LineStyle','None','FaceColor',useColors(2,:)*0.5+0.5);
    xlim([-1.01,1.01]*pi);
    yticks([]);
    ylabel('\theta distribution')
    xticks([-pi,-pi/2,0,pi/2 pi]);
    xticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'});
    meanAng = deg2rad(meanangle([azi1,azi2]));
    %     meanAng = deg2rad((azi1+azi2)/2);
    %     if meanAng<-pi,meanAng = meanAng+2*pi;end
    %     if meanAng>pi,meanAng = meanAng-2*pi;end
    tttemp = get(gca, 'ylim');
    plot([meanAng meanAng],[0 tttemp(2)],'k','LineWidth',2);
    titstr = sprintf('(%s,%s)',aziarraystr{1,ii},aziarraystr{2,ii});
    title(titstr);
    
    htemp = subplot(3,4,ii+4);
    set(htemp,'position',[0.10+0.22*(ii-1),0.41,0.18,0.22])
    HH = [HH,htemp];
    polarhistogramModify(theta(:),thetaUse_1,thetaUse_2,100,useColors);
    %     legend({'1','2'})
    hold on;
    radiustick = get(gca,'RTick');
    meanAng = deg2rad(meanangle([azi1,azi2]));
    polarplot([meanAng meanAng],[0,radiustick(end)],'LineWidth',2,'color','k');
    polarplot([meanAng+pi meanAng+pi],[0,radiustick(end)],'LineWidth',2,'color','k');
    
    
    
    htemp = subplot(3,4,ii+8);
    HH = [HH,htemp];
    thetaShiftUse_1 = theta_shift1(useIndex & mask);
    thetaShiftUse_2 = theta_shift2(useIndex & mask);
    hold on;
    [XX,YY,gmPDF,threshold] = GaussianHalfFit([thetaShiftUse_1,thetaShiftUse_2]);
    contourf(XX,YY,gmPDF,[threshold,inf],'FaceColor',useColors(1,:)*0.5+0.5,'LineColor','k');
    H1 = plot(thetaShiftUse_1(1:50000:end),thetaShiftUse_2(1:50000:end),'*','color',useColors(1,:));
    
    
    thetaShiftUse_1 = theta_shift1(useIndex & ~mask);
    thetaShiftUse_2 = theta_shift2(useIndex & ~mask);
    % find the 50% ofaudio samples that fall int
    [XX,YY,gmPDF,threshold] = GaussianHalfFit([thetaShiftUse_1,thetaShiftUse_2]);
    contourf(XX,YY,gmPDF,[threshold,inf],'FaceColor',useColors(2,:)*0.5+0.5,'LineColor','k');
    H2 = plot(thetaShiftUse_1(1:50000:end),thetaShiftUse_2(1:50000:end),'*','color',useColors(2,:));
    plot([0 1],[0 1],'k','LineWidth',2);
    yticks([0 0.5 1]);
    yticklabels({'0','.5','1'}); text(0.05,0.9,'\chi_2');
    xticks([0 0.5 1]);
    xticklabels({'0','.5','1'}); text(0.85,0.08,'\chi_1');
    
    %     str1 = sprintf('%d%c', azi1, char(176));
    %     str2 = sprintf('%d%c', azi2, char(176));
    str1 = aziarraystr{1,ii};
    str2 = aziarraystr{2,ii};
    legend([H1,H2],{str1,str2},'location','sw');
    box on;
end

for ii = 1:4,
    set(HH((ii-1)*3+1),'position',[0.10+0.22*(ii-1),0.70,0.18,0.26]);
%     set(HH((ii-1)*3+2),'position',[0.10+0.22*(ii-1),0.40,0.24,0.26])
    set(HH((ii-1)*3+3),'position',[0.10+0.22*(ii-1),0.10,0.18,0.26]);
end


a = 0






% calculate the percentage of overlapped theta for two angles from -20 and
% 20
clear all; close all; clc
NFFT = 1024;
Window = 1024;
Shift = 0.25;
load('B_format_RIRs_12BB01_Alfredo_S3A_16k.mat');
[s1,Fs] = audioread('FemaleExample.wav');
[s2,Fs] = audioread('MaleExample.wav');

s1 = s1(1:Fs*10);
s2 = s1(2:Fs*10);
azi1 = 3; %20
azi2 = 35; % -20

sig = s1;
rir = squeeze(rirs(azi1,:,:))';
% apply convolution
p0 = fftfilt(rir(:,1),sig);
vel_x = fftfilt(rir(:,2),sig);
vel_y = fftfilt(rir(:,3),sig);

P0 = stft(p0, NFFT, Window, Shift);
Gx = stft(vel_x, NFFT, Window, Shift);
Gy = stft(vel_y, NFFT, Window, Shift);

Y = real(conj(P0).*Gy);
X = real(conj(P0).*Gx);
theta = atan2(Y,X);

h1 = hist(theta(:),100);

sig = s2; % should we use s1 instead for consistency?
rir = squeeze(rirs(azi2,:,:))';
% apply convolution
p0 = fftfilt(rir(:,1),sig);
vel_x = fftfilt(rir(:,2),sig);
vel_y = fftfilt(rir(:,3),sig);

P0 = stft(p0, NFFT, Window, Shift);
Gx = stft(vel_x, NFFT, Window, Shift);
Gy = stft(vel_y, NFFT, Window, Shift);

Y = real(conj(P0).*Gy);
X = real(conj(P0).*Gx);
theta = atan2(Y,X);

h2 = hist(theta(:),100);

h_overlap = min([h1;h2]);

op_pct = sum(h_overlap)/(sum(h1))











