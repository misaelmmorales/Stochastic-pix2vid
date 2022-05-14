# CNN-RNN Forward Proxy Modeling for CO2 Monitoring

We use the MRST (Matlab Reservoir Simulation Toolbox) to generate high-fidelity simulations for the injection and subsequent migration of CO2 in the SPE 10 model, a benchmark model for reservoir simulation. The top layers of the model represent the Tarbet formation, a prograding near shore environment with Gaussian-distributed rock properties, while the bottom layers represent the Upper Ness formation, a fluvial depositional environment.

The original model is 60 x 220 x 85 cells, with sizes of 20 ft x 10 ft x 2 ft. However, we split generate an ensemble of 2D realizations from the full model by taking each individual layer and splitting it into four 60 x 60 maps in each of the 85 layers. This gives a total of 255 realizations to simulate. 

We run forward simulations using an Automatic Differentiation framework for a period of 5 years injection, monitored monthly (60 timesteps total). The numerical model uses Peaceman's well model, where there is only one injector well in the center of the grid (30,30), and the control is given for a constant injection rate of 5 meters^3/day. The reservoir is originally fully saturated with brine and the injection is pure CO2. With this, we monitor the reservoir pressure and CO2 saturation over time. The results from the high-fidelity simulations are collected along with the rock properties realizations. 

Using this as training data, we fit a CNN-RNN model to learn how to forecast dynamic states from the latent representation of the static reservoir properties (a reduced-order forward model). We perform data augmentation by rotating 90 degrees the feature and target images, for a total of 510 data. This is then split randomly into training and testing sets, and a percentage of traning data is used for validation.The deep learning model is then designed using a block structure that reduces the permeability realizations into a latent space using convolutions, then predicts dynamic latent states using recurrent layers, and then predicts a cube of time-lapse saturation maps in high-resolution. 

This proxy model will allow for more efficient reservoir simulations and possible applications in uncertainty quantification, history matching, and more.

- Gather SPE10 petrphysical properties from MRST spe10, and fluid properties from MRST co2lab
- Generate ensemble of realizations (porosity, permeability) by grid partitioning.
- Perform reservoir simulation to generate dynamic states (pressure, saturation).
- Gather inputs and outputs as <code>.m</code> file, and import into Jupyter notebook using <code>SciPy</code> or <code>HDF5</code>.
- Define the CNN-RNN model
- Compile and Fit using training/validation data
- Verify results using testing data
- Analyze distributions, uncertainty, prior/posterior, computational efficiency, etc.

<p align="center">
  <img src="https://github.com/misaelmmorales/CNN-RNN-Proxy/blob/main/figures/cnn_rnn_architecture.png" width="500" height="250" >
</p>
  
***
