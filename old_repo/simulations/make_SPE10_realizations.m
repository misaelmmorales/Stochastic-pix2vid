%% Script to generate 340 realizations from the SPE10 model
mrstModule add SPE10

% Define layers of interest
layers = 36:85;
 
% Generate true Grid, Wells, and Rock properties
[G_true, W_true, rock_true] = getSPE10setup(layers);

% Define variables
n_layers = G_true.cartDims(3); %total number of layers
dims     = 60;                 %desired subset dimensions

% Predefine temporary porosity/permeability matrices for speed
[all_poro_1, all_poro_2, all_poro_3] = deal(zeros(dims*dims, n_layers));
[all_perm_1, all_perm_2, all_perm_3] = deal(zeros(dims*dims, size(rock_true.perm,2), n_layers));

% Compute individual porosity/permeability realizations and collect
for i=1:n_layers
    rock1 = getSPE10rock(1:dims, 1:60, layers(i));
    all_poro_1(:,i) = rock1.poro;
    all_perm_1(:,:,i) = rock1.perm;

    rock2 = getSPE10rock(1:dims, 61:120, layers(i));
    all_poro_2(:,i) = rock2.poro;
    all_perm_2(:,:,i) = rock2.perm;

    rock3 = getSPE10rock(1:dims, 121:180, layers(i));
    all_poro_3(:,i) = rock3.poro;
    all_perm_3(:,:,i) = rock3.perm;
end

all_poro = cat(2, all_poro_1, all_poro_2, all_poro_3);
all_perm = cat(3, all_perm_1, all_perm_2, all_perm_3);

all_poro = all_poro + 0.01;

all_perm_md = convertTo(all_perm, milli*darcy);

save all_poro all_poro
save all_perm all_perm
save all_perm_md all_perm_md

clear rock1 rock2 rock3 rock4 i
clear all_poro_1 all_poro_2 all_poro_3 all_poro_4
clear all_perm_1 all_perm_2 all_perm_3 all_perm_4