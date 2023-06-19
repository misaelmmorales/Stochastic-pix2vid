function [W] = gen_wells(G, rock, well_loc, ~)
%GEN_WELLS Summary of this function goes here
%   Detailed explanation goes here

num_wells = size(well_loc,1);

%R_inj = 0.1*sum(poreVolume(G,rock))/inj_time;
%R_inj = 500 * stb/day;
%R_inj = (1/num_wells) * 556.2 * 1000 * meter^3 / year; %1 MT/yr
R_inj = (1/num_wells) * 0.5 * 556.2 * 1000 * meter^3 / year; %0.5 MT/yr


W = [];

for i=1:num_wells
    W = verticalWell(W, G, rock, well_loc(i,1), well_loc(i,2), [], ...
                    'Type',          'rate', ...
                    'Val',           R_inj, ...
                    'InnerProduct', 'ip_tpf', ...
                    'Radius',        0.05, ...
                    'Comp_i',        [0 1], ...
                    'name',          ['Injector', int2str(i)] );
end


end