function rt60 = RTsOctave(x,varargin)

%%% This matlab implementation combines the ocative band filtering in
%%% available:  http://www.mathworks.com/matlabcentral/fileexchange/42566-impulse-response-acoustic-information-calculator
%%% and the classic schroeder RT60 calculation algorithm
%%% available:   http://uk.mathworks.com/matlabcentral/fileexchange/35740-blind-reverberation-time-estimation/content/utilities/RT_schroeder.m
%%% Qingju Liu, University of Surrey

% Calculates RT for impulse response in x
%
%   rt60s = RTsOctave(x,varargin)
%
%   rt60 = RTsOctave(x,varargin) returns the reverberation time
%   (to -60 dB). Estimates are taken in octave bands.
%
%
%   ... = IR_stats(...,'parameter',value) allows numerous
%   parameters to be specified. These parameters are:
%       'spec'       : {'mean'} | 'full'
%           Determines the nature of RT60 output. With
%           spec='mean' (default) the reported RT60 is the
%           mean of the 500 Hz and 1 kHz bands. With
%           spec='full', the function returns the RT60 as
%           calculated for each octave band returned in cfs.
%       'region'     : {[-5 -35]} over which the curve is fitted
%       'method'     : {1} | 2  1: least square fitting (default)
%                               2: line between region(1) and (2)
%
%   Octave-band filters are calculated according to ANSI
%   S1.1-1986 and IEC standards. Note that the OCTDSGN
%   function recommends centre frequencies fc in the range
%   fs/200 < fc < fs/5.
%
%   References
%
%   [1] Zahorik, P., 2002: 'Direct-to-reverberant energy
%       ratio sensitivity', The Journal of the Acoustical
%       Society of America, 112, 2110-2117.
%
%   See also OCTDSGN.




% set defaults
options = struct(...
    'graph',false,...
    'spec','mean',...
    'fs',16000,...
    'region', [-5 -35],...
    'method',1);

% read parameter/value inputs
if nargin>1 % if parameters are specified
    % read the acceptable names
    optionNames = fieldnames(options);
    % count arguments
    nArgs = length(varargin);
    if round(nArgs/2)~=nArgs/2
        error('IR_STATS needs propertyName/propertyValue pairs')
    end
    % overwrite defults
    for pair = reshape(varargin,2,[]) % pair is {propName;propValue}
        IX = strcmpi(pair{1},optionNames); % find match parameter names
        if any(IX)
            % do the overwrite
            options.(optionNames{IX}) = pair{2};
        else
            error('%s is not a recognized parameter name',pair{1})
        end
    end
end

% octave-band center frequencies
% The audio spectrum from ~ 20Hz to ~ 20KHz can be divided up into ~ 11 octave bands.
% If we set/define the 7th octave band’s center frequency to be f7 = 1000Hz,
% then all lower and higher center frequencies for octave bands can be de found Conversely,
cfs = [31.25 62.5 125 250 500 1000 2000 4000 8000 16000];

% octave-band filter order
N = 3;

% read in impulse
fs = options.fs;
assert(fs>=5000,'Sampling frequency is too low. FS must be at least 5000 Hz.')


% get number of channels
numchans = size(x,2);

% limit centre frequencies so filter coefficients are stable
cfs = cfs(cfs>fs/200 & cfs<fs/5);
cfs = cfs(:);

% calculate filter coefficients
a = zeros(length(cfs),(2*N)+1);
b = zeros(length(cfs),(2*N)+1);
for f = 1:length(cfs)
    [b(f,:),a(f,:)] = octdsgn(cfs(f),fs,N);
end

% empty matrices to fill
rt = zeros([length(cfs) numchans]);

% filter and integrate
for n = 1:numchans
    t0 = find(x(:,n).^2==max(x(:,n).^2)); % find direct impulse
    if options.graph
        scrsz = get(0,'ScreenSize');
        figpos = [((n-1)/numchans)*scrsz(3) scrsz(4) scrsz(3)/2 scrsz(4)];
        figure('Name',['Channel ' num2str(n)],'OuterPosition',figpos);
    end
    for f = 1:length(cfs)
        y = filter(b(f,:),a(f,:),x(:,n)); % octave-band filter
        %%
        %[rt,EDC_log] = RT_schroeder(y(:)',fs,region,1,0);
        region=options.region;
        [RT,EDC_log,yy] = RT_schroeder1(y(:)',fs,region,options.method,0);
        rt(f,n)=RT;
        
        if options.graph % plot
            plotcurve(y,yy,RT,EDC_log,region,fs,n,t0,cfs,f);
        end
    end
    
end

switch lower(options.spec)
    case 'full'
        rt60 = rt;
    case 'mean'
        % rt60 = mean(rt); % overall RT60
        rt60 = mean(rt(cfs==500 | cfs==1000,:)); % overall selected RT60
    otherwise
        error('Unknown ''spec'': must be ''full'' or ''mean''.')
end



function plotcurve(y,yy,RT,EDC_log,region,fs,n,t0,cfs,f)

linewidth = 1;
fontsize = 10;
% time axes for different vectors
ylength = length(y);
ty = ((0:ylength-1)-t0(n))./fs;
y_length = length(yy);
t_y = ((0:y_length-1)-t0(n))./fs;
%             tE_rt = (0:length(E_rt)-1)./fs;
%             tE_edt = (0:length(E_edt)-1)./fs;
% plot
subplot(length(cfs),2,(2*f)-1)
plot(ty,y,'k') % octave-band impulse
if f==1
    title({'Impulse response'; ''; [num2str(cfs(f)) ' Hz octave band']})
else
    title([num2str(cfs(f)) ' Hz octave band'])
end
if f==length(cfs)
    xlabel('Time [s]')
else
    set(gca,'xticklabel',[]);
end
ylabel('Amplitude')
set(gca,'position',[1 1 1 1.05].*get(gca,'position'),'xlim',[min(ty) max(ty)]);
subplot(length(cfs),2,2*f)
% energy decay and linear least-square fit
plot(ty,EDC_log,'LineWidth',linewidth);hold on;
plot(t_y,yy,'-r','LineWidth',linewidth);
line( [0 ylength/fs],[region(1)  region(1)],'Color','black','LineStyle','--','LineWidth',linewidth);
line( [0 ylength/fs],[region(2)  region(2)],'Color','black','LineStyle','--','LineWidth',linewidth);
line( [0 ylength/fs],[-60 -60],'Color','black','LineStyle','--','LineWidth',linewidth);
axis([0 ylength/fs -65 0]);
set(gca,'fontsize',fontsize);
% title for top row
if f==1
    title({'Decay curve'; ''; [num2str(cfs(f)) ' Hz octave band']})
else
    title([num2str(cfs(f)) ' Hz octave band'])
end
% x label for bottom row
if f==length(cfs)
    xlabel('Time [s]')
else
    set(gca,'xticklabel',[]);
end
ylabel('Energy [dB]')
set(gca,'position',[1 1 1 1.05].*get(gca,'position'),'ylim',[-70 0],'xlim',[0 ylength/fs]);
%title('Reverberation time estimation - Schroeder method','FontSize',fontsize);
text(.5*ylength/fs ,-20,['T_6_0 = ',num2str(RT, '%.2f'), 's'],'FontSize',fontsize,'color','k');



function [RT,EDC_log,y] = RT_schroeder1(h,fs,region,method,delay_comp)
%--------------------------------------------------------------------------
% Measuring the Reverberation Time using the Schroeder Method
%--------------------------------------------------------------------------
%
% Input:      h:  inpulse response
%             fs: sampling frequency in [Hz]
%             Optional:
%                region(2): Region in the EDC where the RT is computed [dB]
%                           default [-5 -35]
%                method:    detection method
%                            1: least square fitting (default)
%                            2: line between region(1) and (2)
%                delay_comp: 0: no delay compensation (default)
%                            1: compensate sound propagation delay
%
% Output:     RT: reveberation time in [s]
%             EDC_log: energy decay curve (log normalized)
%             t_EDC: time vector for EDC curve
%             noise_floor: noise floor
%             (at time instance the fitting curve reaches -60dB)
%--------------------------------------------------------------------------
%
% Copyright (c) 2012, Marco Jeub and Heinrich Loellmann
% Institute of Communication Systems and Data Processing
% RWTH Aachen University, Germany
% Contact information: jeub@ind.rwth-aachen.de
%
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the RWTH Aachen University nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%--------------------------------------------------------------------------


if nargin < 5, delay_comp = 0; end
if nargin < 4, method = 1; end
if nargin < 3, region = [-5,-35]; end

if nargin < 2; error('at least the RIR and its sampling frequency must be given');end;
%--------------------------------------------------------------------------

h_length=length(h);
%--------------------------------------------------------------------------
% Compensate sound propagation
% (exclude parts of the RIR before the direct path)
%--------------------------------------------------------------------------
%
if delay_comp == 1
    [~,prop_delay]=max(h);
    h(1:h_length-prop_delay+1)=h(prop_delay:h_length);
    h_length=length(h);
end

%--------------------------------------------------------------------------
% Energy decay curve
%--------------------------------------------------------------------------
h_sq=h.^2;
h_sq = fliplr(h_sq);
EDC = cumsum(h_sq);
EDC = fliplr(EDC);

% normalize to '1'
EDC_norm=EDC./max(EDC);

%--------------------------------------------------------------------------
% Estimate the reverberation time
%--------------------------------------------------------------------------
if method == 1  % least square fitting
    % first value of the EDC decaying 60dB (10^-6)
    EDC_log = 10*log10(EDC_norm);
    %EDC_60 = find (EDC_log <= -60, 1, 'first');
    EDC_reg1  = find (EDC_log <= region(1), 1, 'first');
    EDC_reg2 = find (EDC_log <= region(2), 1, 'first');
    
    EDC_reg12 = EDC_log(EDC_reg1:EDC_reg2);
    x=1:length(EDC_reg12);
    p = polyfit(x,EDC_reg12,1); % linear least square fitting
    
    x=1:length(EDC_reg12);
    y=p(1)*x+p(2);
    
    y0=y(1)-p(1)*EDC_reg1;
    
    % intersection of polyfit line with -60dB
    x_rt = (-60-y0)/p(1);
    
    RT=x_rt/fs;   % Reverberation time in [s]
    
    % fitting line from 0 to -60dB
    x=1:x_rt;
    y = p(1)*x+y0;
end;
%--------------------------------------------------------------------------
if method == 2  % simple line between the 2 thresholds in region
    % first value of the EDC decaying 60dB (10^-6)
    EDC_log = 10*log10(EDC_norm);
    %EDC_60 = find (EDC_log <= -60, 1, 'first');
    EDC_reg1  = find (EDC_log <= region(1), 1, 'first');
    EDC_reg2 = find (EDC_log <= region(2), 1, 'first');
    
    % Line Slope between the Points given by region(1) and region(2)
    m=( EDC_log(EDC_reg2) - EDC_log(EDC_reg1)) / ( EDC_reg2 - EDC_reg1);
    % Line
    x=1:h_length;
    y=EDC_log(EDC_reg1)+m*(x-EDC_reg1);
    % point of intersection at -60dB
    x2=(-60-y(1)) / m;
    
    RT=x2/fs;   % Reverberation time in [s]
    
end

% [EOF]

