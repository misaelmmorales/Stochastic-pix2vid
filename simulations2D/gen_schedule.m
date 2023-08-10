%function [schedule, dT1, dT2] = gen_schedule(W, bc, dT1, dT2)
function[schedule, dT1] = gen_schedule(W, bc, dT1)
%GEN_SCHEDULE Summary of this function goes here
%   Detailed explanation goes here

% Control Schedule (injection period only)
schedule.control = struct('W', W, 'bc', bc);
schedule.step.val = dT1;
schedule.step.control = ones(numel(dT1),1);

%{
nwells = size(W,1);
% Schedule Control (injection period - monitoring period)
schedule.control    = struct('W', W, 'bc', bc);
schedule.control(2) = struct('W', W, 'bc', bc);
for i=1:nwells
    schedule.control(2).W(i).val = 0;
end
% Schedule Step
schedule.step.val = [dT1; dT2];
schedule.step.control = [ones(numel(dT1),1) ; ones(numel(dT2),1)*2];
%}

end

