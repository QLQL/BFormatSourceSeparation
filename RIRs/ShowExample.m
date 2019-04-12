% Plot one example of the mixture, groundtruth and separated results for
% 2-source conditions

clear all; close all; clc

sample_i = 126;
source_i = 1; % show the first signal
SourceNumber = 2;
data_dir = sprintf('/vol/vssp/ucdatasets/s3a/Qingju/EusipcoEvalResult/Mixture%d/',SourceNumber)
taglegArray = {{'Oracle IBM','IBM'},
    {'Proposed',sprintf('FeatureProposedNsrc%dWei',SourceNumber)},
    {'Spec+\theta\_U',sprintf('FeatureSpecSpatNsrc%dWei',SourceNumber)},
    {'Spec-only\_U',sprintf('FeatureSpecOnlyNsrc%dWei',SourceNumber)},
    {'Spec+\chi\_Chimera',sprintf('ChimeraNsrc%dWei',SourceNumber)},
    {'Chen et al','GMMNew'}
    };

tagArray = cell(1,length(taglegArray)); % the signal appendix
legArray = cell(1,length(taglegArray)); % legend
for algorithm=1:length(taglegArray),
    temp = taglegArray{algorithm};
    tagArray{algorithm} = temp{2};
    legArray{algorithm} = temp{1};
end


figure('position',[1200,10,600,980]);
for sample_i = 10 % [10:50]
    disp(["======================================",num2str(sample_i)]);
%     pause;
    try
        %% read in all the signals
        sourceName = sprintf('Ind_%d_src%d.wav',sample_i,source_i-1);
        mixName = sprintf('Ind_%d_mix.wav',sample_i);
        source = audioread([data_dir,sourceName]);
        mix = audioread([data_dir,mixName]);
        source = source./(max(abs(source))*5); % normalisation
        mix = mix./(max(abs(mix))*5); % normalisation
        %         figure;plot(source);hold on;plot(mix+0.2);
        NFFT = 1024;
        hopsize = 0.25;
        [sigLogPower,~] = logFeatureExtraction(source,NFFT,hopsize);
        [mixLogPower,mixAngle] = logFeatureExtraction(mix,NFFT,hopsize);
        
        EstArray = cell(1,length(tagArray));
        EstSpecArray = cell(1,length(tagArray));
        for algorithm = 1:length(tagArray),
            tag = tagArray{algorithm};
            estimateName = sprintf('Ind_%d_%s_est%d.wav',sample_i,tag,source_i-1);
            
            [Estimate,Fs] = audioread([data_dir,estimateName]);
            Estimate = Estimate./(max(abs(Estimate))*5); % normalisation
            EstArray{algorithm} = Estimate;
            
            [temp,~] = logFeatureExtraction(Estimate,NFFT,hopsize);
            EstSpecArray{algorithm} = temp;
            
            L = min(length(source),length(Estimate));
            %% SI score
            stoival = stoi(source(1:L),Estimate(1:L),Fs);
            Results(1,algorithm) = stoival;
            fprintf('The current STOI is %f\n',stoival);
            
            
            
            %% PESQ score
            pesqval = pesq(Estimate(1:L),source(1:L),16000);
            Results(2,algorithm) = pesqval;
            fprintf('The current PESQ is %f \n',pesqval);
            
            
            
            %% SDR score
            sdrval=eval_sdr(Estimate(1:L),source(1:L),Fs);
            Results(3,algorithm) = sdrval;
            fprintf('The current SDR is %f\n',sdrval);
            
            %     max(abs(Estimate))
        end
        
        
        IndexPlot = 1:180;
        IndexPlot1 = [80:130];%[80:130];
        IndexPlot2 = [10:270];%[10:500];
        
        %% plot mixture and groundtruth
        axisrange = [-25,5];
        h = axes('position',[0.10 0.82 0.75 0.17]);
        %tt = max(max(mixLogPower(:)));
        tt=0;
        pcolor(mixLogPower(:,IndexPlot)-tt)
        shading interp
        YTick = [1 64 128 192 256]*2;
        set(gca,'YTick',YTick)
        set(gca,'YTickLabel',{'0','2','4','6','8'});
        set(gca,'XTickLabel',[]);
        % new white bounding box on top
        caxis(axisrange)
        colormap(jet)
        set(gca,'TickLength',[0.015,0.015]);
        
        hold on
        temp1 = [IndexPlot1(1)-IndexPlot(1),IndexPlot1(end)-IndexPlot(1),IndexPlot1(end)-IndexPlot(1),IndexPlot1(1)-IndexPlot(1),IndexPlot1(1)-IndexPlot(1)];
        temp2 = [IndexPlot2(1)+2,IndexPlot2(1)+2,IndexPlot2(end)+2,IndexPlot2(end)+2,IndexPlot2(1)+2];
        plot(temp1,temp2,'k','linewidth',2)
        text(10,450,'Mixture','FontSize',14,'FontWeight','bold');
        
        
        
        set(gca,'LineWidth',2,'Layer','top')
        ax2 = axes('Position', get(h, 'Position'),'Color','none');
        set(ax2,'LineWidth',2,'XTick',[],'YTick',[],'XColor','w','YColor','w','box','on','layer','top')
        
        
        
        h = axes('position',[0.10 0.64 0.75 0.17]);
        % tt = max(max(sigLogPower(:)));
        tt=0;
        % pcolor(sigLogPower(IndexPlot2,IndexPlot1)-tt )
        pcolor(sigLogPower(:,IndexPlot)-tt)
        shading interp
        caxis(axisrange)
        hold on
        plot(temp1,temp2,'k','linewidth',2)
        text(10,450,'Groundtruth','FontSize',14,'FontWeight','bold');
        YTick = [1 64 128 192]*2;
        set(gca,'YTick',YTick)
        set(gca,'YTickLabel',{'0','2','4','6'});
        set(gca,'XTickLabel',get(gca,'XTick')*0.016);
        % xlabel('Time (sec)')
        colorbar('position',[0.87 0.64 0.05 0.35])
        set(gca,'TickLength',[0.015,0.015]);
        
        set(gca,'LineWidth',2,'Layer','top')
        ax2 = axes('Position', get(h, 'Position'),'Color','none');
        set(ax2,'LineWidth',2,'XTick',[],'YTick',[],'XColor','w','YColor','w','box','on','layer','top')
        ylabel('Frequency (kHz)','color','k','pos',[-0.025    1    1],'FontSize',14); % Create label
        
        
        %% plot all algorithms
        
        
        XTick = [10,30];
        YTick = [1,32];
        W = 0.4;
        H = 0.18;
        S = [0.10,0.52];
        Y = [0.43,0.24,0.05];
        for algorithm = 1:6,
            row = ceil(algorithm/2);
            column = 2-mod(algorithm,2);
            h = axes('position',[S(column) Y(row) W H]);
            
            tt = 0;
            tempSpec = EstSpecArray{algorithm};
            pcolor(tempSpec(IndexPlot2,IndexPlot1)-tt )
            shading interp
            if row==3,
                set(gca,'XTick',[2,22,42]);
                set(gca,'XTickLabel',{'1.28','1.6','1.92'});
                xlabel('Time (sec)','FontSize',14)
            else
                set(gca,'XTick',[2,22,42]);
                set(gca,'XTickLabel',[]);
            end
            
            if column==1,
                %                 set(gca,'YTick',[128,256,384]);
                %                 set(gca,'YTickLabel',{'2','4','6'});
                set(gca,'YTick',[1,128,256]);
                set(gca,'YTickLabel',{'0','2','4'});
                set(gca,'YTick',[1,128,256]);
                set(gca,'TickLength',[0.02,0.02]);
            else
                %                 set(gca,'YTick',[128,256,384]);
                set(gca,'YTick',[1,128,256]);
                set(gca,'YTickLabel',[]);
                set(gca,'TickLength',[0.02,0.02]);
            end
            caxis(axisrange)
            % text(5,450,legArray{algorithm},'FontSize',14,'FontWeight','bold');
            % text(5,400,sprintf('%0.2f~~%0.2f~~%0.2fdB',Results(2,algorithm),Results(1,algorithm),Results(3,algorithm)),'FontSize',10,'FontWeight','bold');
            text(5,240,legArray{algorithm},'FontSize',14,'FontWeight','bold');
            text(5,200,sprintf('%0.2f~~%0.2f~~%0.2fdB',Results(2,algorithm),Results(1,algorithm),Results(3,algorithm)),'FontSize',10,'FontWeight','bold');
            set(gca,'LineWidth',2,'Layer','top')
            ax2 = axes('Position', get(h, 'Position'),'Color','none');
            set(ax2,'LineWidth',2,'XTick',[],'YTick',[],'XColor','w','YColor','w','box','on','layer','top')
            
        end
    catch
        disp("----\n")
    end
end

a = 0


