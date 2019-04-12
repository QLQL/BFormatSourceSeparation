function [sigLogPower,sigAngle] = logFeatureExtraction(sig,NFFT,HopSize_or_HopPercent)
if nargin<3,HopSize_or_HopPercent = 0.5;end
if nargin<2,NFFT = 256;end

sigf= stft(sig, NFFT, NFFT, HopSize_or_HopPercent);
sigLogPower = log(abs(sigf).^2);
sigAngle = angle(sigf);