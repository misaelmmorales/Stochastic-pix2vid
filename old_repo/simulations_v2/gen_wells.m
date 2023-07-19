function [W] = gen_wells(G, rock)
%GEN_WELLS Summary of this function goes here
%   Detailed explanation goes here

I_inj = G.cartDims(1)/2;
J_inj = G.cartDims(2)/2;
R_inj = 50 * meter^3 / day;

W = [];
W = verticalWell(W, G, rock, I_inj, J_inj, [],...
                'Type', 'rate', ...
                'Val', R_inj, ...
                'InnerProduct', 'ip_tpf', ...
                'Radius', 0.05, ...
                'Comp_i', [0 1], ...
                'name', 'Injector');



end

