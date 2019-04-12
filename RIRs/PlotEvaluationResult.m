% plot evaluations results.
clear all; close all; clc


randn('seed',123456789);
rand('seed',123456789);

figure('Position', [800, 100, 500, 600]);
SourceNumberArray = [2,3];


for sourceNumi = 1:2
    SourceNumber = SourceNumberArray(sourceNumi);
    
    data_dir = sprintf('/vol/vssp/ucdatasets/s3a/Qingju/EusipcoEvalResult/Mixture%d/',SourceNumber);
    
    taglegArray = {{'Oracle IBM','IBM'},
        {'Proposed',sprintf('FeatureProposedNsrc%dWei',SourceNumber)},
%         {'ProposedBinary',sprintf('FeatureProposedNsrc%dWeiBinary',SourceNumber)},
%         {'ProposedSoft',sprintf('FeatureProposedNsrc%dWeiSoft',SourceNumber)},
        {'Spec+\theta\_U',sprintf('FeatureSpecSpatNsrc%dWei',SourceNumber)},
%         {'Spec+\thetaSoft',sprintf('FeatureSpecSpatNsrc%dWeiSoft',SourceNumber)},
        {'Spec-only\_U',sprintf('FeatureSpecOnlyNsrc%dWei',SourceNumber)},
        {'Spec+\chi\_Chimera',sprintf('ChimeraNsrc%dWei',SourceNumber)},
        {'Chen et al','GMMNew'}
        };
    
    
    %         {'GMM [18]','GMM'},
    
    tagArray = cell(1,length(taglegArray)); % the signal appendix
    legArray = cell(1,length(taglegArray)); % legend
    for algorithm=1:length(taglegArray),
        temp = taglegArray{algorithm};
        tagArray{algorithm} = temp{2};
        legArray{algorithm} = temp{1};
    end
    
    if sourceNumi==1,
        PlotResult_mean = zeros(length(tagArray),3,2);
        PlotResult_std = PlotResult_mean;
    end
    
    % The following few lines is for statiscal significance validation via ttest
    saveName = sprintf('FeatureProposedNsrc%dWei',SourceNumber);
    load(saveName)
    pesq_proposed = PESQResults;
    stoi_proposed = SIResults;
    sdr_proposed = SDRResults;
    % The above few lines is for statiscal significance validation via ttest
    
    for algorithm = 1:length(tagArray),
        %     PESQResults = zeros(N,SourceNumber);
        %     SIResults = zeros(N,SourceNumber);
        %     SDRResults = zeros(N,SourceNumber);
        tag = tagArray{algorithm};
        fprintf('========================================= Algorithm %s\n',tag);
        
        saveName = sprintf('%s.mat',tag);
        if strcmp(tag,'IBM') || strcmp(tag(1:3),'GMM') || strcmp(tag,'Shujau'),
            saveName = sprintf('%sNsrc%d.mat',tag,SourceNumber);
        end
        %         if strcmp(tag(1:3),'GMM'),
        %             saveName = sprintf('%sNsrc%d.mat',tag,SourceNumber);
        %         end
        load(saveName);
        
        % calculate the average result and mean
        for metric=1:size(PlotResult_mean,2)
            switch metric
                case 1,
                    temp = PESQResults;
                    tempproposed = pesq_proposed;
                case 2
                    temp = SIResults;
                    tempproposed = stoi_proposed;
                case 3
                    temp = SDRResults;
                    tempproposed = sdr_proposed;
            end
            
            [h,p,ci,stats] = ttest2(temp(:),tempproposed(:))
            
            useIndex = 1:length(temp)*SourceNumber;
            % adding some threshold to remove outliers
            if strcmp(tag(1:3),'GMM') || strcmp(tag,'Shujau'),
                useIndex = find(temp>-5);
            elseif length(tag)>15
                if strcmp(tag(1:15),'FeatureSpecOnly')
                    useIndex = find(temp>-5);
                end
            end
            tempMean = mean(temp(useIndex));
            tempStd = std(temp(useIndex));
            fprintf('====%f====%f\n',tempMean,tempStd);
            
            PlotResult_mean(algorithm, metric, sourceNumi) = tempMean;
            PlotResult_std(algorithm, metric, sourceNumi) = tempStd;
            
        end
        
        
    end
    
end

% % paired t-test
% SourceNumber = 2;
% saveName = sprintf('IBMNsrc%d.mat',SourceNumber);
% load(saveName)
% pesq_ibm = SIResults;
% saveName = sprintf('FeatureProposedNsrc%dWei',SourceNumber);
% load(saveName)
% pesq_proposed = SIResults;
% [h,p,ci,stats] = ttest(pesq_ibm(:),pesq_proposed(:))

hh = [];
for metric=1:size(PlotResult_mean,2)
    meanV = squeeze(PlotResult_mean(:, metric, :))';
    varV = squeeze(PlotResult_std(:, metric, :))';
    
    hh = [hh,subplot(size(PlotResult_mean,2),1,metric)];
%     h1 = bar(meanV);
    hold on;
    for i = 1:size(meanV,2),
        if i==1,
            b = bar([1,2]-0.52+0.134*i,meanV(:,i),'BarWidth',0.10);
            set(b,'FaceColor',[1,1,1])
        else
            b = bar([1,2]-0.465+0.134*i,meanV(:,i),'BarWidth',0.10);
            if i==2,
                set(b,'FaceColor',[1,1,1]*0.7)
            end
        end
    end
%     set(h1(1),'XData',get(h1(1),'XData')-0.1)
    %     xlim([0.5 4.5])
    
    
    %     xlabel('Input SNR (dB)','fontsize',14)
    %     legend(path_array,'Location','northwest');
    %     ylabel('PESQ','fontsize',14)
    ax = gca;
    ax.ColorOrderIndex = 1;
    for i = 1:size(meanV,2),
        if i==1,
            e = errorbar([1,2]-0.52+0.134*i,meanV(:,i),zeros(1,2),varV(:,i));
        else
            e = errorbar([1,2]-0.465+0.134*i,meanV(:,i),zeros(1,2),varV(:,i));
        end
        
        %         e.LineWidth = 2;
        e.LineStyle = 'none';
        e.Color = 'k';
    end
    hold off
    set(gca,'XTick',[1,2])
    xlim([0.5,2.5])
    
    switch metric
        case 1
            ylabel('PESQ (-0.5-4.5)','fontsize',12);
            ylim([0 3.5]);
            set(gca,'xaxisLocation','top')
            set(gca,'xticklabel',{'Two-source','Three-source'},'fontsize',12);
            for i = 1:size(meanV,2),
                    leg = legArray{i};
                    if i==1,
                        h=text(1-0.52+0.134*i,0.1,leg);
                    else
                        h=text(1-0.465+0.134*i,0.1,leg);
                    end
                    set(h,'Rotation',90);
                    set(h,'FontName','Arial','FontWeight','bold');
            end
            
        case 2
            ylabel('STOI (0-1)','fontsize',12);
            ylim([0.4 1]);
            set(gca,'YTick',0.5:0.1:0.9)
            set(gca,'XTickLabel',[]);
            for i = 1:size(meanV,2),
                    leg = legArray{i};
                    if i==1,
                        h=text(1-0.52+0.134*i,0.415,leg);
                 	else
                        h=text(1-0.465+0.134*i,0.415,leg);
                    end
                    set(h,'Rotation',90);
                    set(h,'FontName','Arial','FontWeight','bold');
            end
        case 3
            ylabel('SDR (dB)','fontsize',12);
            ylim([0 18.5]);
            set(gca,'XTickLabel',{'{\it I}=2','{\it I}=3'},'fontsize',12);
            for i = 1:size(meanV,2),
                    leg = legArray{i};
                    if i==1,
                        h=text(1-0.52+0.134*i,0.4,leg);
                    else
                        h=text(1-0.465+0.134*i,0.4,leg);
                    end
                    set(h,'Rotation',90);
                    set(h,'FontName','Arial','FontWeight','bold');
            end
    end
    box on
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
end

set(hh(1),'position',[0.15,0.64,0.82,0.28])
set(hh(2),'position',[0.15,0.35,0.82,0.28])
set(hh(3),'position',[0.15,0.06,0.82,0.28])





