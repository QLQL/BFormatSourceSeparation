

clear all; close all; clc


N = 200;

% fileName = 'Proposed2Src1.csv'; % the first 40 epochs
% M = csvread(fileName,1);
% TrainLoss_DC = M(1:80,3);
% TrainLoss_Mask = M(1:80,5);
% ValidLoss_DC = M(1:80,8);
% ValidLoss_Mask = M(1:80,10);
%
% fileName = 'Proposed2Src2.csv'; % continue training
% M = csvread(fileName,1);
% TrainLoss_DCtemp = [TrainLoss_DC;M(81:N,3)];
% TrainLoss_Masktemp = [TrainLoss_Mask;M(81:N,5)];
% ValidLoss_DCtemp = [ValidLoss_DC;M(81:N,8)];
% ValidLoss_Masktemp = [ValidLoss_Mask;M(81:N,10)];





algorithmArray = {'Proposed','SpecSpat','SpecOnly','Chimera'};
path_array = {'Proposed','Spec+\theta\_U','Spec-only\_U','Spec+\chi\_Chimera'};

trainvalidFlag = 0; % 1 train 0 validation

Loss_all = zeros(N,4,length(algorithmArray),2);
NsrcArray = [2,3];
for Nsrci = 1:2
    Nsrc = NsrcArray(Nsrci);
    for algi = 1:length(algorithmArray),
        fileName = sprintf('%sNsrc%dlog.csv',algorithmArray{algi},Nsrc);
        if exist(fileName),
            M = csvread(fileName,1);
            Loss_all(:,1,algi) = M(1:N,3); % TrainLoss_DC
            Loss_all(:,2,algi) = M(1:N,5); %TrainLoss_Mask
            Loss_all(:,3,algi) = M(1:N,8); %ValidLoss_DC
            Loss_all(:,4,algi) = M(1:N,10);%ValidLoss_Mask
        else
            Loss_all(:,1,algi) = TrainLoss_DCtemp; % TrainLoss_DC
            Loss_all(:,2,algi) = TrainLoss_Masktemp; %TrainLoss_Mask
            Loss_all(:,3,algi) = ValidLoss_DCtemp; %ValidLoss_DC
            Loss_all(:,4,algi) = ValidLoss_Masktemp;%ValidLoss_Mask
        end
        
    end
end




myColors = [0.7 0.7 0.7;
    0.9290 0.6940 0.1250;
    0.4940 0.1840 0.5560;
    0.4660 0.6740 0.1880];
% legend boxoff
% set(hleg, 'position',[.18 .52 .3 .3])

EpochN = 200;
linewidth = 2;


figure('position',[500,500,450,250]);hold on;

clear all; close all; clc


N = 200;

% fileName = 'Proposed2Src1.csv'; % the first 40 epochs
% M = csvread(fileName,1);
% TrainLoss_DC = M(1:80,3);
% TrainLoss_Mask = M(1:80,5);
% ValidLoss_DC = M(1:80,8);
% ValidLoss_Mask = M(1:80,10);
%
% fileName = 'Proposed2Src2.csv'; % continue training
% M = csvread(fileName,1);
% TrainLoss_DCtemp = [TrainLoss_DC;M(81:N,3)];
% TrainLoss_Masktemp = [TrainLoss_Mask;M(81:N,5)];
% ValidLoss_DCtemp = [ValidLoss_DC;M(81:N,8)];
% ValidLoss_Masktemp = [ValidLoss_Mask;M(81:N,10)];





algorithmArray = {'Proposed','SpecSpat','SpecOnly','Chimera'};
path_array = {'Proposed','Spec +\theta\_U','Spec-only\_U','Spec +\chi\_Chimera'};

trainvalidFlag = 0; % 1 train 0 validation

Loss_all = zeros(N,4,length(algorithmArray),2);
NsrcArray = [2,3];
for Nsrci = 1:2
    Nsrc = NsrcArray(Nsrci);
    for algi = 1:length(algorithmArray),
        fileName = sprintf('%sNsrc%dlog.csv',algorithmArray{algi},Nsrc);
        if exist(fileName),
            M = csvread(fileName,1);
            Loss_all(:,1,algi,Nsrci) = M(1:N,3); % TrainLoss_DC
            Loss_all(:,2,algi,Nsrci) = M(1:N,5); %TrainLoss_Mask
            Loss_all(:,3,algi,Nsrci) = M(1:N,8); %ValidLoss_DC
            Loss_all(:,4,algi,Nsrci) = M(1:N,10);%ValidLoss_Mask
            %         else
            %             Loss_all(:,1,algi) = 0; % TrainLoss_DC
            %             Loss_all(:,2,algi) = 0; %TrainLoss_Mask
            %             Loss_all(:,3,algi) = 0; %ValidLoss_DC
            %             Loss_all(:,4,algi) = 0;%ValidLoss_Mask
        end
        
    end
end




myColors = [0.5 0.5 0.5;
    0.9290 0.6940 0.1250;
    0.4940 0.1840 0.5560;
    0.4660 0.6740 0.1880];
% legend boxoff
% set(hleg, 'position',[.18 .52 .3 .3])

EpochN = 200;
linewidth = 2;


%% plot1
figure('position',[500,500,450,250]);hold on;



if trainvalidFlag,
    Loss_DC_all = squeeze(Loss_all(:,1,:,:));
    Loss_Mask_all = squeeze(Loss_all(:,2,:,:));
else
    Loss_DC_all = squeeze(Loss_all(:,3,:,:));
    Loss_Mask_all = squeeze(Loss_all(:,4,:,:));
end


for Nsrci = 1:2
    Nsrc = NsrcArray(Nsrci);
    hh = [];
    Loss_DC = squeeze(Loss_DC_all(:,:,Nsrci));
    Loss_Mask = squeeze(Loss_Mask_all(:,:,Nsrci));
    
    for algi = 1:length(algorithmArray),
        if Nsrci==1,
            % htemp = plot(Loss_DC(:,algi)+Loss_Mask(:,algi),'-','linewidth',linewidth,'color',myColors(algi,:));
            htemp = plot(Loss_DC(:,algi),'-','linewidth',linewidth,'color',myColors(algi,:));
            hh = [hh,htemp];
        else
            % plot(Loss_DC(:,algi)+Loss_Mask(:,algi),'--','linewidth',1,'color',myColors(algi,:));
            plot(Loss_DC(:,algi),'--','linewidth',1,'color',myColors(algi,:));
        end
        
    end
    
    %     aa = Loss_DC(150:end,:)+Loss_Mask(150:end,:);
    %     bb = mean(aa);
    %     %     (bb(2)-bb(1))/bb(1) % compare proposed with spec+spat
    %     %     (bb(4)-bb(1))/bb(1) % compare proposed with Chimera
    %     mean(Loss_Mask(190:end,3)) % mask error for spec-only
    
end

hold off
hleg = legend(hh,path_array);
% legend boxoff
% set(hleg, 'position',[.18 .52 .3 .3])
% axis tight
xlim([0 EpochN])
box on
% ylim([0.1 0.55])
xlabel('Epoch')
ylabel('DC loss')


%% plot2
figure('position',[500,500,450,250]);hold on;



if trainvalidFlag,
    Loss_DC_all = squeeze(Loss_all(:,1,:,:));
    Loss_Mask_all = squeeze(Loss_all(:,2,:,:));
else
    Loss_DC_all = squeeze(Loss_all(:,3,:,:));
    Loss_Mask_all = squeeze(Loss_all(:,4,:,:));
end


for Nsrci = 1:2
    Nsrc = NsrcArray(Nsrci);
    hh = [];
    Loss_DC = squeeze(Loss_DC_all(:,:,Nsrci));
    Loss_Mask = squeeze(Loss_Mask_all(:,:,Nsrci));
    
    for algi = 1:length(algorithmArray),
        if Nsrci==1,
            htemp = plot(Loss_Mask(:,algi),'-','linewidth',linewidth,'color',myColors(algi,:));
            hh = [hh,htemp];
        else
            plot(Loss_Mask(:,algi),'--','linewidth',1,'color',myColors(algi,:));
        end
        
    end
    
    %     aa = Loss_DC(150:end,:)+Loss_Mask(150:end,:);
    %     bb = mean(aa);
    %     %     (bb(2)-bb(1))/bb(1) % compare proposed with spec+spat
    %     %     (bb(4)-bb(1))/bb(1) % compare proposed with Chimera
    %     mean(Loss_Mask(190:end,3)) % mask error for spec-only
    
end

hold off
hleg = legend(hh,path_array);
% legend boxoff
% set(hleg, 'position',[.18 .52 .3 .3])
% axis tight
xlim([0 EpochN])
box on
% ylim([0.1 0.55])
xlabel('Epoch')
ylabel('Mask loss')

