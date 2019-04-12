% Apply evaluations to separated signals.
clear all; close all; clc


randn('seed',123456789);
rand('seed',123456789);

SourceNumber = 3;
data_dir = sprintf('/DirectoryToYourEstimatedData/Mixture%d/',SourceNumber);

% taglegArray = {{'Proposed',sprintf('FeatureProposedNsrc%dWei',SourceNumber)},
%     {'IBM','IBM'},
%     {'Spec+\theta',sprintf('FeatureSpecSpatNsrc%dWei',SourceNumber)},
%     {'Spec-only',sprintf('FeatureSpecOnlyNsrc%dWei',SourceNumber)},
%     {'Chimera [6]',sprintf('ChimeraNsrc%dWei',SourceNumber)},
%     {'Chen','GMMNew'}
%     };

% taglegArray = {{'GMM','GMM'},{'Chen','GMMNew'}};
% taglegArray = {{'Shujau','Shujau'}};
taglegArray = {{'Spec+\thetaSoft',sprintf('FeatureSpecSpatNsrc%dWeiSoft',SourceNumber)}};
% taglegArray = {
%     {'ProposedBinary',sprintf('FeatureProposedNsrc%dWeiBinary',SourceNumber)},
%     {'ProposedSoft',sprintf('FeatureProposedNsrc%dWeiSoft',SourceNumber)}};

tagArray = cell(1,length(taglegArray)); % the signal appendix
legArray = cell(1,length(taglegArray)); % legend
for algorithm=1:length(taglegArray),
    temp = taglegArray{algorithm};
    tagArray{algorithm} = temp{2};
    legArray{algorithm} = temp{1};
end

N = 200;
for algorithm = 1:length(tagArray),
    PESQResults = zeros(N,SourceNumber);
    SIResults = zeros(N,SourceNumber);
    SDRResults = zeros(N,SourceNumber);
    tag = tagArray{algorithm};
    fprintf('========================================= Algorithm %s\n',tag);

    for sample_i = 1:N
        fprintf('******* Algorithm %s ******i* Mixture %d \n',tag,sample_i);

        for source_i = 1:SourceNumber,
            sourceName = sprintf('Ind_%d_src%d.wav',sample_i,source_i-1);
            estimateName = sprintf('Ind_%d_%s_est%d.wav',sample_i,tag,source_i-1);

            [source,Fs] = audioread([data_dir,sourceName]);
            [Estimate,Fs] = audioread([data_dir,estimateName]);

            %             figure;
            %             starti = Fs*1;endi = Fs*1.5;
            %             plot(source(starti:endi));
            %             hold on;
            %             plot(Estimate(starti:endi));
            
            if length(Estimate)<length(source),
                source = source(1:length(Estimate));
            end
            

            %% SI score
            try
                stoival = stoi(source,Estimate,Fs);
            catch
                stoival = -100;
                disp('The current STOI calculation failed');
            end
            SIResults(sample_i,source_i) = stoival;
            fprintf('The current STOI is %f\n',stoival);



            %% PESQ score
            try
                pesqval = pesq(Estimate,source,16000);
            catch
                pesqval = -100;
                disp('The current PESQ calculation failed');
            end
            PESQResults(sample_i,source_i) = pesqval;
            fprintf('The current PESQ is %f \n',pesqval);



            %% SDR score
            try
                sdrval=eval_sdr(Estimate,source,Fs);
            catch
                sdrval = -100;
                disp('The current SDR calculation failed');
            end
            SDRResults(sample_i,source_i) = sdrval;
            fprintf('The current SDR is %f\n',sdrval);


        end

    end
    
    saveName = sprintf('%s.mat',tag);
    if strcmp(tag,'IBM') || strcmp(tag(1:3),'GMM') || strcmp(tag,'Shujau'),
        saveName = sprintf('%sNsrc%d.mat',tag,SourceNumber);
    end
    %     if strcmp(tag(1:3),'GMM'),
    %         saveName = sprintf('%sNsrc%d.mat',tag,SourceNumber);
    %     end
    save(saveName,'PESQResults','SIResults', 'SDRResults');
end









% % Distribution of the mixture in angle difference.
% clear all; close all; clc
% 
% 
% randn('seed',123456789);
% rand('seed',123456789);
% 
% SourceNumber = 2;
% data_dir = sprintf('/DirectoryToYourEstimatedData/Mixture%d/',SourceNumber);
% 
% filename2 = 'TIMITtestingSignalList.mat';
% testData = load(filename2);
% chosenAngsList = testData.chosenAngsList;
% thetaNorm = testData.thetaNorm;
% rirs = testData.rirs;
% clear testData
% 
% 
% % First plot the distributions of input angles for two source conditions
% AngleDiff = diff(chosenAngsList');
% AngleDiff(AngleDiff>180) = AngleDiff(AngleDiff>180)-360;
% AngleDiff(AngleDiff<-180) = AngleDiff(AngleDiff<-180)+360;
% AngleDiff = abs(AngleDiff);
% % edges = 35:10:185;
% % figure('position',[400 200 500 200])
% % histogram(AngleDiff,edges,'Normalization','probability');
% % xlabel('Input azimuth difference [deg]');
% % ylabel('PDF')
% 
% edges = [35,85,135,185];
% 
% [N,~] = histcounts(AngleDiff,edges);
% use1 = AngleDiff<edges(2); use1 = [use1(:),use1(:)];
% use2 = edges(2)<AngleDiff & AngleDiff<edges(3); use2 = [use2(:),use2(:)];
% use3 = edges(3)<AngleDiff & AngleDiff<edges(4); use3 = [use3(:),use3(:)];
% 
% fprintf('There are respective %d---%d---%d mixtures \n for the angle difference between 40-80, 90-130, 140-180 degrees \n', N(1),N(2),N(3))
% 
% 
% 
% SourceNumber = 2;
% 
% data_dir = sprintf('/DirectoryToYourEstimatedData/Mixture%d/',SourceNumber);
% 
% taglegArray = {{'IBM','IBM'},
%     {'Proposed',sprintf('FeatureProposedNsrc%dWei',SourceNumber)},
%     {'Spec+\theta',sprintf('FeatureSpecSpatNsrc%dWei',SourceNumber)},
%     {'Spec-only',sprintf('FeatureSpecOnlyNsrc%dWei',SourceNumber)},
%     {'Chimera [6]',sprintf('ChimeraNsrc%dWei',SourceNumber)},
%     {'Chen','GMMNew'}
%     };
% 
% tagArray = cell(1,length(taglegArray)); % the signal appendix
% legArray = cell(1,length(taglegArray)); % legend
% for algorithm=1:length(taglegArray),
%     temp = taglegArray{algorithm};
%     tagArray{algorithm} = temp{2};
%     legArray{algorithm} = temp{1};
% end
% 
% PlotResult_mean = zeros(length(tagArray),3,3);
% 
% 
% for algorithm = 1:length(tagArray),
%     %     PESQResults = zeros(N,SourceNumber);
%     %     SIResults = zeros(N,SourceNumber);
%     %     SDRResults = zeros(N,SourceNumber);1
%     tag = tagArray{algorithm};
%     fprintf('========================================= Algorithm %s\n',tag);
%     
%     saveName = sprintf('%s.mat',tag);
%     if strcmp(tag,'IBM'),
%         saveName = sprintf('%sNsrc%d.mat',tag,SourceNumber);
%     end
%     if strcmp(tag(1:3),'GMM'),
%         saveName = sprintf('%sNsrc%d.mat',tag,SourceNumber);
%     end
%     load(saveName);
%     
%     % calculate the average result and mean
%     for metric=1:size(PlotResult_mean,2)
%         switch metric
%             case 1,
%                 temp = PESQResults;
%             case 2
%                 temp = SIResults;
%             case 3
%                 temp = SDRResults;
%         end
%         
%         useIndex = true(200,2);
%         if strcmp(tag,'GMM')
%             useIndex = temp>-5;
%         elseif length(tag)>15
%             if strcmp(tag(1:15),'FeatureSpecOnly')
%                 useIndex = temp>-5;
%             end
%         end
%         
%         use = find(use1 & useIndex);
%         tempMean1 = mean(temp(use));
%         
%         use = find(use2 & useIndex);
%         tempMean2 = mean(temp(use));
%         
%         use = find(use3 & useIndex);
%         tempMean3 = mean(temp(use));
%         
% %         fprintf('=====%f=====%f=====%f=====\n',tempMean1,tempMean2,tempMean3);
%         
%         PlotResult_mean(algorithm, metric,:) = [tempMean1,tempMean2,tempMean3];
%         
%     end
%     squeeze(PlotResult_mean(algorithm, :,:))'
%     
%     
% end
