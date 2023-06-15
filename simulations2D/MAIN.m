%% Main variables
clear;clc;close all
set(0,'DefaultFigureWindowStyle','docked')

% Import MRST module
mrstModule add SPE10 co2lab
mrstModule add ad-core ad-props ad-blackoil mrst-gui

% Define global variables
dims = 128;

%% Make Grid
nx=dims;       ny=dims;       nz=1; 
dx=1000*meter; dy=1000*meter; dz=50*meter; 

% Make cartesian grid
G = cartGrid([nx ny nz], [dx dy dz]);
G = computeGeometry(G);

%% Make Initial State
gravity on;  g = gravity;
rhow = 1000; % density of brine corresponding to 94 degrees C and 300 bar
%initState.pressure = rhow * g(3) * G.cells.centroids(:,3);
initState.pressure = G.cells.centroids(:,3) * 3000 * psia;
initState.s = repmat([1, 0], G.cells.num, 1);
initState.sGmax = initState.s(:,2);

%% Make Fluid
co2     = CO2props();             % load sampled tables of co2 fluid properties
p_ref   = 30 * mega * Pascal;     % choose reference pressure
t_ref   = 94 + 273.15;            % choose reference temperature, in Kelvin
rhoc    = co2.rho(p_ref, t_ref);  % co2 density at ref. press/temp
cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc; % co2 compressibility
cf_wat  = 0;                      % brine compressibility (zero)
cf_rock = 4.35e-5 / barsa;        % rock compressibility
muw     = 8e-4 * Pascal * second; % brine viscosity
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

% Use function 'initSimpleADIFluid' to make a simple fluid object
fluid = initSimpleADIFluid('phases', 'WG'           , ...
                           'mu'  , [muw, muco2]     , ...
                           'rho' , [rhow, rhoc]     , ...
                           'pRef', p_ref            , ...
                           'c'   , [cf_wat, cf_co2] , ...
                           'cR'  , cf_rock          , ...
                           'n'   , [2 2]);

% Modify relative permeability curves
srw = 0.27;
src = 0.20;
fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));

% Add capillary pressure
pe = 5 * kilo * Pascal;
pcWG = @(sw) pe * sw.^(-1/2);
fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5)); 

%% Make Boundary Conditions
bc = [];
vface_ind = (G.faces.normals(:,3) == 0);
bface_ind = (prod(G.faces.neighbors, 2) == 0);
bc_face_ix = find(vface_ind & bface_ind);
bc_cell_ix = sum(G.faces.neighbors(bc_face_ix, :), 2);
p_face_pressure = initState.pressure(bc_cell_ix);
bc = addBC(bc, bc_face_ix, 'pressure', p_face_pressure, 'sat', [1,0]);

%% Define Timesteps
timestep1  = rampupTimesteps(5*year, year/6, 0);
timestep2  = rampupTimesteps(45*year, year, 0);
total_time = [timestep1; timestep2];
%total_time = timestep1;

%% Generate Models & Run Simulation
N = 1000;
M = size(total_time,1); %number of schedule timesteps (75)

parfor i=1:2
    fprintf('Simulation %i\n', i)
    rock                     = gen_rock(i)
    W                        = gen_wells(G, rock)
    [schedule, dT1, dT2]     = gen_schedule(W, bc, timestep1, timestep2)
    %[schedule, dT1]          = gen_schedule(W, bc, timestep1)
    [model, wellSol, states] = gen_simulation(G, rock, fluid, initState, schedule)


    result{i} = states;
end

%% Collect and Export Results
for i=1:N
    for j=1:M
        sol(i,j)          = result{1,i}{j,1};  %define results cell as struct
        pressure(i,:,j)   = sol(i,j).pressure; %collect pressure states
        saturation(i,:,j) = sol(i,j).s(:,2);   %collect saturation states
    end
end

for i=1:N
    poro(i,:) = all_poro(:,i);
    perm(i,:) = all_perm_md(:,1,i);
end

save pressure pressure
save saturation saturation
save poro poro
save perm perm

%% Make Plots
% figure
% for i=1:9
%     subplot(3,3,i)
%     plotCellData(G, states{i*10}.s(:,2))
%     time = convertTo(cumsum(total_time), year);
%     axis tight; title([num2str(time(i*10)), ' years'])
%     view(25,65);

%% END