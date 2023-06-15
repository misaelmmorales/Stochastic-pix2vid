function [W] = gen_wells(G, rock, well_loc)
%GEN_WELLS Summary of this function goes here
%   Detailed explanation goes here
I_inj = well_loc(1);
J_inj = well_loc(2);
R_inj = 500 * stb/day;

W = [];
W = verticalWell(W, G, rock, I_inj, J_inj, [],...
                'Type',          'rate', ...
                'Val',           R_inj, ...
                'InnerProduct', 'ip_tpf', ...
                'Radius',        0.05, ...
                'Comp_i',        [0 1], ...
                'name',          'Injector');



end

