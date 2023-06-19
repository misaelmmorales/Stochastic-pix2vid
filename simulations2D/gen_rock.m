function [rock] = gen_rock(realization)
%GEN_ROCK Summary of this function goes here
%   Detailed explanation goes here

facies_name = sprintf('facies/facies%d.mat', realization);
poro_name = sprintf('porosity/poro%d.mat', realization);
perm_name = sprintf('permeability/logperm%d.mat', realization);

facies_dat = load(fullfile(pwd(), facies_name)).facies;
poro_dat   = load(fullfile(pwd(), poro_name)).poro;
perm_dat   = load(fullfile(pwd(), perm_name)).logperm;

poro_sample = facies_dat .* poro_dat;
perm_sample = facies_dat .* perm_dat;
perm_fin   = 10.^perm_sample*milli*darcy;

rock.poro = poro_sample';
rock.perm = perm_fin';

end

