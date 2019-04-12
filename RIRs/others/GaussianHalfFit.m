function [XX,YY,gmPDF,threshold] = GaussianHalfFit(data)

% find the 50% ofaudio samples that fall into the ellips
newUseIndex = true(size(data,1),1);
for iter = 1:3,
    GMModel = fitgmdist([data(newUseIndex,1),data(newUseIndex,2)],1);
    gmPDF = pdf(GMModel,data);
    tempThreshold = median(gmPDF);
    newUseIndex = gmPDF>tempThreshold;
end


x = linspace(0,1,50);
y = linspace(0,1,50);
[XX,YY] = meshgrid(x,y);
gmPDF = pdf(GMModel,[XX(:) YY(:)]);
gmPDF = reshape(gmPDF,50,50);

%     temp = sort(gmPDF(:),'ascend');
%     bb = cumsum(temp);
%     [~,ind] = min(abs(bb-bb(end)*0.3));
%     threshold  = temp(ind);
threshold = tempThreshold;

%     contour(XX,YY,gmPDF,[threshold,threshold],[colors(1,:),'-'],'LineWidth',2)