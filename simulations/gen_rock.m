function [rock] = gen_rock(all_poro, all_perm, realization)
%GEN_ROCK Summary of this function goes here
%   Detailed explanation goes here

rock.poro = all_poro(:,realization);
rock.perm = all_perm(:,:,realization);

end

