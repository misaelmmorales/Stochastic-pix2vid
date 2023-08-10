function [rock] = gen_rock(realization, perm, poro, facies)
%GEN_ROCK Summary of this function goes here
%   Detailed explanation goes here

perm_dat = perm(:,realization);
poro_dat = poro(:,realization);
facies_dat = facies(:,realization);

poro_sample = facies_dat .* poro_dat;
perm_sample = facies_dat .* perm_dat;
perm_fin   = 10.^perm_sample*milli*darcy;

rock.poro = poro_sample;
rock.perm = perm_fin;

end

