function x = istft(d, Window_or_WindowSize, HopSize_or_HopPercent)
% Inverse short-time Fourier transform.
% NFFT should be a number power to 2.
% Qingju Liu

size_d = size(d);
NFFT = (size_d(1)-1)*2;
if nargin<2, Window_or_WindowSize = NFFT;end
if nargin<3, HopSize_or_HopPercent=.25;end
if length(Window_or_WindowSize)==1,
    Window=ones(Window_or_WindowSize,1); % default is the rectangle window.
else
    Window=Window_or_WindowSize(:);
end
W = length(Window);

if HopSize_or_HopPercent>1,
    HopSize = round(HopSize_or_HopPercent);
else
    HopSize = fix(W.*HopSize_or_HopPercent);
end


cols = size_d(2);
xlen = W + (cols-1)*HopSize;
x = zeros(1,xlen);

d =[d;conj(d(end-1:-1:2,:))];
Seg = real(ifft(d));
if W<=NFFT,
    WindowedSeg = Seg(1:W,:).*repmat(Window(:),1,cols);
else
    WindowedSeg = [Seg;zeros(W-NFFT-1,cols)].*repmat(Window(:),1,cols);
end

for block = 1:cols,
    temp = WindowedSeg(:,block);
    x((block-1)*HopSize+(1:W)) = x((block-1)*HopSize+(1:W)) + temp(:)';
end

end