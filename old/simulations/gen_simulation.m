function [model, wellSol, states] = gen_simulation(G, rock, fluid, initState, schedule)
%GEN_SIMULATION Summary of this function goes here
%   Detailed explanation goes here

model  = TwoPhaseWaterGasModel(G, rock, fluid);
[wellSol, states] = simulateScheduleAD(initState, model, schedule);
% 'NonLinearSolver', NonLinearSolver()

end

