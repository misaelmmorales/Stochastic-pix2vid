%function [schedule, dT1, dT2] = gen_schedule(W, bc, dT1, dT2)
function[schedule, dT1] = gen_schedule(W, bc, dT1)
%GEN_SCHEDULE Summary of this function goes here
%   Detailed explanation goes here

%{
% Control Schedule (injection period - monitoring period)
schedule.control = struct('W', W, 'bc', bc);
schedule.control(2) = struct('W', W, 'bc', bc);
schedule.control(2).W.val = 0;
% schedule
schedule.step.val = [dT1; dT2];
schedule.step.control = [ones(numel(dT1),1) ; ones(numel(dT2),1)*2];
%}

% Control Schedule (injection period - monitoring period)
schedule.control = struct('W', W, 'bc', bc);

% schedule
schedule.step.val = dT1;
schedule.step.control = ones(numel(dT1),1);

end

