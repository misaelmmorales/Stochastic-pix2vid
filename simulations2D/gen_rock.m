function [rock] = gen_rock(realization)
%GEN_ROCK Summary of this function goes here
%   Detailed explanation goes here

facies_name = sprintf('facies/facies%d.mat', realization);
poro_name = sprintf('porosity/porosity%d.mat', realization);
perm_name = sprintf('permeability/logperm%d.mat', realization);

facies_dat = load(fullfile(pwd(), facies_name)).facies'+0.66;
poro_dat   = load(fullfile(pwd(), poro_name)).poro;
perm_dat   = 10.^load(fullfile(pwd(), perm_name)).logperm *milli*darcy;

poro_sample = facies_dat .* poro_dat;
perm_sample = (10.^(facies_dat .* log10(convertTo(perm_dat,milli*darcy))))*milli*darcy;

perm(:,1) = perm_sample;
perm(:,2) = perm_sample;
perm(:,3) = 0.1*perm_sample;

rock.poro = poro_sample;
rock.perm = perm;

end

