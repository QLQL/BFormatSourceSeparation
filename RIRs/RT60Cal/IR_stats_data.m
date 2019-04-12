function [rt60,drr,cte,cfs,edt] = IR_stats_data(x,varargin)
% Calculate RT, DRR, Cte, and EDT for impulse response file
% 
%   rt60 = IR_stats(filename)
%   [rt60,drr] = IR_stats(filename)
%   [rt60,drr,cte] = IR_stats(filename)
%   [rt60,drr,cte,cfs] = IR_stats(filename)
%   [rt60,drr,cte,cfs,edt] = IR_stats(filename)
%   ... = IR_stats(...,'parameter',value)
% 
%   rt60 = IR_stats(filename) returns the reverberation time
%   (to -60 dB) using a method based on ISO 3382-1:2009.
%   The function uses reverse cumulative trapezoidal
%   integration to estimate the decay curve, and a linear
%   least-square fit to estimate the slope between 0 dB and
%   -60 dB. Estimates are taken in octave bands and the
%   overall figure is an average of the 500 Hz and 1 kHz
%   bands.
% 
%   The function attempts to identify the direct impulse as
%   the peak of the squared impulse response.
% 
%   filename should be the full path to a wave file or the
%   name of a wave file on the Matlab search path. The file
%   can have any number of channels, estimates (and plots)
%   will be returned for each channel.
% 
%   [rt60,drr] = IR_stats(filename) returns the DRR for the
%   impulse. This is calculated in the following way:
%   
%   DRR = 10 * log10( x(t0-c:t0+c)^2 / x(t0+c+1:end)^2 )
% 
%   where x is the approximated integral of the impulse, t0
%   is the time of the direct impulse, and c=2.5ms [1].
% 
%   [rt60,drr,cte] = IR_stats(filename) returns the
%   early-to-late index Cte for the impulse. This is
%   calculated in the following way:
%   
%   Cte = 10 * log10( x(t0-c:t0+te)^2 / x(t0+te+1:end)^2 )
% 
%   where x is the approximated integral of the impulse and
%   te is a point 50 ms after the direct impulse.
% 
%   [rt60,drr,cte,cfs] = IR_stats(filename) returns the
%   octave-band centre frequencies used in the calculations.
% 
%   [rt60,drr,cte,cfs,edt] = IR_stats(filename) returns the
%   early decay time (EDT). The slope of the decay curve is
%   determined from the fit between 0 and -10 dB. The decay
%   time is calculated from the slope as the time required
%   for a 60 dB decay.
% 
%   ... = IR_stats(...,'parameter',value) allows numerous
%   parameters to be specified. These parameters are:
%       'graph'      : {true} | false
%           Controls whether decay curves are plotted.
%           Specifically, graphs are plotted of the impulse
%           response, decay curves, and linear least-square
%           fit(s) for each octave band and channel of the
%           wave file.
%       'te'         : {0.05} | scalar
%           Specifies the early time limit (in seconds).
%       'spec'       : {'mean'} | 'full'
%           Determines the nature of RT60 output. With
%           spec='mean' (default) the reported RT60 is the
%           mean of the 500 Hz and 1 kHz bands. With
%           spec='full', the function returns the RT60 as
%           calculated for each octave band returned in cfs.
%       'y_fit'      : {[0 60]} | two-element vector
%           Specifies the decibel range over which the decay
%           curve should be evaluated. For example, 'y_fit'
%           may be [-5 -25] or [-5 -35] corresponding to the
%           RT20 and RT30 respectively.
%       'correction' : {0.0025} | scalar
%           Specifies the correction parameter c (in
%           seconds) given above for DRR and Cte
%           calculations. Values of up to 10 ms have been
%           suggested in the literature.
% 
%   Octave-band filters are calculated according to ANSI
%   S1.1-1986 and IEC standards. Note that the OCTDSGN
%   function recommends centre frequencies fc in the range
%   fs/200 < fc < fs/5.
% 
%   The author would like to thank Feifei Xiong for his
%   input on the correction parameter.
% 
%   References
% 
%   [1] Zahorik, P., 2002: 'Direct-to-reverberant energy
%       ratio sensitivity', The Journal of the Acoustical
%       Society of America, 112, 2110-2117.
% 
%   See also OCTDSGN.

% !---
% ==========================================================
% Last changed:     $Date: 2014-09-03 11:42:54 +0100 (Wed, 03 Sep 2014) $
% Last committed:   $Revision: 306 $
% Last changed by:  $Author: ch0022 $
% ==========================================================
% !---


% set defaults
options = struct(...
    'graph',false,...
    'te',0.05,...
    'spec','mean',...
    'y_fit',[0 -60],...
    'correction',0.0025);

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
cfs = [31.25 62.5 125 250 500 1000 2000 4000 8000 16000];

% octave-band filter order
N = 3;

% read in impulse
fs = 16000;
assert(fs>=5000,'Sampling frequency is too low. FS must be at least 5000 Hz.')

% set te in samples
te = round(options.te*fs);

% Check sanity of te
assert(te<length(x),'The specified early time limit te is longer than the duration of the impulse!')

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
z = zeros([length(cfs) size(x)]);
rt = zeros([length(cfs) numchans]);
edt = zeros([length(cfs) numchans]);
t0 = zeros(1,numchans);
drr = zeros(1,numchans);
cte = zeros(1,numchans);

correction = round(options.correction*fs);

% filter and integrate
for n = 1:numchans
    t0(n) = find(x(:,n).^2==max(x(:,n).^2)); % find direct impulse
    if options.graph
        scrsz = get(0,'ScreenSize');
        figpos = [((n-1)/numchans)*scrsz(3) 0 scrsz(3)/2 scrsz(4)];
        figure('Name',['Channel ' num2str(n)],'OuterPosition',figpos);
    end
    for f = 1:length(cfs)
        y = filter(b(f,:),a(f,:),x(:,n)); % octave-band filter
        temp = cumtrapz(y(end:-1:1).^2); % decay curve
        z(f,:,n) = temp(end:-1:1);
        [edt(f,n),E_edt,fit_edt] = calc_decay(z(f,t0:end,n),[0,-10],60,fs); % estimate EDT
        try
            use_y_fit = options.y_fit;
            [rt(f,n),E_rt,fit_rt] = calc_decay(z(f,t0:end,n),use_y_fit,60,fs); % estimate RT
        catch
            try
                use_y_fit = [options.y_fit(1), options.y_fit(2)+10];
                [rt(f,n),E_rt,fit_rt] = calc_decay(z(f,t0:end,n),use_y_fit,60,fs); % estimate RT
            catch
                use_y_fit = [options.y_fit(1), options.y_fit(2)+20];
                [rt(f,n),E_rt,fit_rt] = calc_decay(z(f,t0:end,n),use_y_fit,60,fs); % estimate RT
            end
        end
        
        while abs(rt(f,n)-edt(f,n))>0.5,
            use_y_fit(2) = use_y_fit(2)+10
            [rt(f,n),E_rt,fit_rt] = calc_decay(z(f,t0:end,n),use_y_fit,60,fs); % estimate RT
        end
        if options.graph % plot
            % time axes for different vectors
            ty = ((0:length(y)-1)-t0(n))./fs;
            tE_rt = (0:length(E_rt)-1)./fs;
            tE_edt = (0:length(E_edt)-1)./fs;
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
            if nargout==5
                % plot EDT fit if EDT wanted
                plot(tE_rt,E_rt,'-k',tE_rt,fit_rt,'--r',tE_edt,fit_edt,':b')
            else
                plot(tE_rt,E_rt,'-k',tE_rt,fit_rt,'--r')
            end
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
            set(gca,'position',[1 1 1 1.05].*get(gca,'position'),'ylim',[-70 0],'xlim',[0 max(tE_rt)]);
            % choose legend according to EDT request
            if nargout==5
                legend('Energy decay curve',['Linear fit (RT' num2str(abs(diff(use_y_fit))) ')'],'Linear fit (EDT)','location','northeast')
            else
                legend('Energy decay curve','Linear fit (RT60)','location','northeast')
            end
        end
    end
    % DRR
    if nargout>=2
        drr(n) = 10.*log(...
            trapz(x(max(1,t0(n)-correction):t0(n)+correction,n).^2)/...
            trapz(x(t0(n)+correction+1:end,n).^2)...
            );
    end
    % Cte
    if nargout>=3
        if t0(n)+te+1>size(x,1)
            warning(['Early time limit (te) out of range in channel ' num2str(n) '. Try lowering te.'])
            cte(n) = NaN;
        else
            cte(n) = 10.*log(...
                trapz(x(max(1,t0(n)-correction):t0(n)+te).^2)/...
                trapz(x(t0(n)+te+1:end,n).^2)...
                );
        end
    end
end

switch lower(options.spec)
    case 'full'
        rt60 = rt;
    case 'mean'
        rt60 = mean(rt(cfs==500 | cfs==1000,:)); % overall RT60
        edt = mean(edt(cfs==500 | cfs==1000,:)); % overall EDT
    otherwise
        error('Unknown ''spec'': must be ''full'' or ''mean''.')
end

% ----------------------------------------------------------
% Local functions:
% ----------------------------------------------------------

% ----------------------------------------------------------
% calc_decay: calculate decay time from decay curve
% ----------------------------------------------------------

function [t,E,fit] = calc_decay(z,y_fit,y_dec,fs)
% Returns the time for a specified decay y_dec calculated
% from the fit over the range y_fit. The input is the
% integral of the impulse sample at fs Hz. The function also
% returns the energy decay curve in dB and the corresponding
% fit.

E = 10.*log10(z); % put into dB
E = E-max(E); % normalise to max 0
E = E(1:find(isinf(E),1,'first')-1); % remove trailing infinite values
IX = find(E<=max(y_fit),1,'first'):find(E<=min(y_fit),1,'first'); % find yfit x-range

% calculate fit over yfit
x = reshape(IX,1,length(IX));
y = reshape(E(IX),1,length(IX));
p = polyfit(x,y,1);
fit = polyval(p,1:2*length(E)); % actual fit
fit2 = fit-max(fit); % fit anchored to 0dB

diff_y = abs(diff(y_fit)); % dB range diff
t = (y_dec/diff_y)*find(fit2<=-diff_y,1,'first')/fs; % estimate decay time

fit = fit(1:length(E));

% [EOF]