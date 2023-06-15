function [W] = gen_wells(G, rock, well_loc, inj_time)
%GEN_WELLS Summary of this function goes here
%   Detailed explanation goes here
I_inj = well_loc(1);
J_inj = well_loc(2);

%R_inj = 0.1*sum(poreVolume(G,rock))/inj_time;
%R_inj = 500 * stb/day;

R_inj = 556.2 *1000 * meter^3 / year; %1MT/yr

W = [];
W = verticalWell(W, G, rock, I_inj, J_inj, [],...
                'Type',          'rate', ...
                'Val',           R_inj, ...
                'InnerProduct', 'ip_tpf', ...
                'Radius',        0.05, ...
                'Comp_i',        [0 1], ...
                'name',          'Injector');



end