function [W] = gen_wells(G, rock)
%GEN_WELLS Summary of this function goes here
%   Detailed explanation goes here

I_inj = 64;
J_inj = 64;
R_inj = 1.5e4 * meter^3 / day;

W = [];
W = verticalWell(W, G, rock, I_inj, J_inj, [],...
                'Type',          'rate', ...
                'Val',           R_inj, ...
                'InnerProduct', 'ip_tpf', ...
                'Radius',        0.05, ...
                'Comp_i',        [0 1], ...
                'name',          'Injector');



end

