# HippoCoMM.V1
A 3D neuroinspired computational model of the hippocampus CA1 subfield

Tutorial for the HippoComm Software.

# Generalities

When you open the software, it will look like this:

<img src="img/image1.png" alt="Figure" width="600"/>

It has various different views that will be explained later.

# Control panel
<img src="img/image2.png" alt="Figure">

This part is used to set the desired tissue dimensions.

The user must enter at least two columns.

 <img src="img/image3.png" alt="Figure"/> will compute the x, y and z from neuron density and the number of cell.

 <img src="img/image4.png" alt="Figure"/> will compute the neuron density from x, y, and z and the number of cell.

 <img src="img/image5.png" alt="Figure"/> will compute the number of cell from x, y, and z and the neuron density.

<img src="img/image6.png" alt="Figure"/>This part determines the distribution of neurons among the different types (PYR, BAS, OLM, and BIS).

<img src="img/image7.png" alt="Figure"/> The top part is expressed as a percentage. The sum of the four numbers on the left must be 100; otherwise, the Sum column will turn red


<img src="img/image8.png" alt="Figure"/> to show the user that a value needs to be updated.

Notice that PYRSTIM, which are PYR cells from CA3 that project onto PYR cells from CA1, are expressed as a percentage of the total number of PYR cells.

<img src="img/image9.png" alt="Figure"/> This will compute the number of cells in each group, which is shown in the bottom line:

 <img src="img/image10.png" alt="Figure"/>

# Interconnection between neurons

<img src="img/image11.png" alt="Figure"/>

Here we set the interconnection between each type of neurons.

If the number N (integer) is >=1, this means that the source will project onto N neurons

If the number N (float) is 0<N<1, this means that only a fraction of the neurons will project onto one neuron. For instance:  
If N = 0.5, this means that only 1 neuron out of 2 will project onto 1 neuron.  
If N = 0.2, this means that only 1 neuron out of 5 will project onto 1 neuron.

# Intercellular typical distance

<img src="img/image12.png" alt="Figure"/>

Here is a typical distance between neuron types. It is based on the literature. This table is used when connections are randomly chosen while still taking the distance between cells into account.

# Cell Placement

<img src="img/image13.png" alt="Figure"/>

Here, the user can select different way to place the neuron on the 3D environment.

 <img src="img/image14.png" alt="Figure"/>cells are place on a 2D plan grid (x,y) equally spaced

<img src="img/image15.png" alt="Figure"/>

<img src="img/image16.png" alt="Figure"/> cells are placed on a 2D plan grid (x,y) with equal spacing, but the PYR cell from CA3 are isolated.

<img src="img/image17.png" alt="Figure"/>

<img src="img/image18.png" alt="Figure"/> Cells are placed on a 3D grid (x, y, z) with equal spacing.

<img src="img/image19.png" alt="Figure"/>

<img src="img/image20.png" alt="Figure"/> Cells are placed on a 3D grid (x, y, z) with equal spacing, but the PYR cells from CA3 are isolated.

<img src="img/image21.png" alt="Figure"/>

<img src="img/image22.png" alt="Figure"/> Cells are placed randomly on a 2D plane (x, y).

<img src="img/image23.png" alt="Figure"/>

<img src="img/image24.png" alt="Figure"/> Cells are placed on a 2D grid (x, y), but the PYR cells from CA3 are isolated.  

<img src="img/image25.png" alt="Figure"/>

<img src="img/image26.png" alt="Figure"/> Cells are placed randomly on a 3D plane (x, y, z).

<img src="img/image27.png" alt="Figure"/>

<img src="img/image28.png" alt="Figure"/> Cells are placed randomly on a 3D plane (x, y, z), but the PYR cells from CA3 are isolated. 

<img src="img/image29.png" alt="Figure"/>

<img src="img/image30.png" alt="Figure"/> Cells are placed randomly on a 2D plane (x, y). This is the Best Candidate algorithm.

<img src="img/image31.png" alt="Figure"/>

<img src="img/image32.png" alt="Figure"/> Cells are placed randomly on a 2D plane (x, y). This is the Best Candidate algorithm, but the PYR cells from CA3 are isolated.

<img src="img/image33.png" alt="Figure"/>

<img src="img/image34.png" alt="Figure"/> Cells are placed randomly on a 3D plane (x, y, z). This is the Best Candidate algorithm. 

<img src="img/image35.png" alt="Figure"/>

<img src="img/image36.png" alt="Figure"/> Cells are placed randomly on a 3D plane (x, y, z). This is the Best Candidate algorithm, but the PYR cells from CA3 are isolated. 

<img src="img/image37.png" alt="Figure"/>

Notice that each time the user places cells in the 3D view, the connections between neurons are reinitialized. This means that you change the position of the cells in space as well as the interconnections between them.

<img src="img/image38.png" alt="Figure"/>

This allows you to change the scale of the tissue without replacing the cells or their interconnections.

# Stimulation Parameters

Each stimulation is a pulse apply onto a PYR cell from CA3.

Here the user can tuned the different parameters for the stimulation.

<img src="img/image39.png" alt="Figure"/>

<img src="img/image40.png" alt="Figure"/> pulse duration <img src="img/image41.png" alt="Figure"/>


.<img src="img/image42.png" alt="Figure"/> Pulse amplitude

<img src="img/image43.png" alt="Figure"/>Time constant used to smooth the pulse

0: <img src="img/image44.png" alt="Figure"/>  5: <img src="img/image45.png" alt="Figure"/>

<img src="img/image46.png" alt="Figure"/> how much pulse will be applied

<img src="img/image47.png" alt="Figure"/> Variance from the center at which to place the pulses:

5: 
<img src="img/image48.png" alt="Figure"/>

20 : 
<img src="img/image49.png" alt="Figure"/>

50 :
<img src="img/image50.png" alt="Figure"/>

<img src="img/image51.png" alt="Figure"/>The seed is used to force exactly the same noise to be added to the simulation if the number is different from 0. If the number is 0, the simulations will be different for each run.

<img src="img/image52.png" alt="Figure"/> There are two types of stimulation: ‘Burst’, when the stimulation is centered in the middle of the stimulation duration, and ‘Continuous’, which generates continuous stimulation throughout the entire stimulation duration.

<img src="img/image53.png" alt="Figure"/>


<img src="img/image54.png" alt="Figure"/> duration of each simulation

<img src="img/image55.png" alt="Figure"/> sampling rate

<img src="img/image56.png" alt="Figure"/> Display or hide the VTK 3D environment if it is not necessary.

<img src="img/image57.png" alt="Figure"/> Display or hide the temporal signal view if it is not necessary.

<img src="img/image58.png" alt="Figure"/> If the number of temporal signals to plot is too large, only the specified percentage of signals will be plotted.

<img src="img/image59.png" alt="Figure"/> Create the cortical model according to the cell placement. Be careful: each time the cell placement is changed, the cortical model must be recreated (not necessary if you have used the <img src="img/image60.png" alt="Figure"/> button)

# Update Model’s Parameters

<img src="img/image61.png" alt="Figure"/> This will display an interface where all parameters can be tuned. 

<img src="img/image62.png" alt="Figure"/>

Every parameter of each cell is available, but it can be tedious to use.

<img src="img/image63.png" alt="Figure"/> This allows you to modify a group of cells at once.

An interface will appear:

<img src="img/image64.png" alt="Figure"/>

<img src="img/image65.png" alt="Figure"/> The user can select the nearest cell from a given position.

<img src="img/image66.png" alt="Figure"/> The user can select a cell by its ID.

<img src="img/image67.png" alt="Figure"/> The user can select a cell directly by clicking on the 3D view.

<img src="img/image68.png" alt="Figure"/> Once selected, the parameters of the selected cell will appear in the left list.

The user may change the values displayed here and apply them to all selected cells (this will be explained later when selecting targeted cells).

<img src="img/image69.png" alt="Figure"/> This will save the current parameters to a file (parameters from the list).

<img src="img/image70.png" alt="Figure"/> This will load a previously saved file (it must correspond to the type of the selected cell).

To select other cells, use the rightmost panel. Notice that only cells of the same type as the selected cell (on the left) can be selected. Selected cells will have a white box around them to make them distinguishable.

<img src="img/image71.png" alt="Figure"/> Unselect all cells.

<img src="img/image72.png" alt="Figure"/> Select all cells of the same type as the currently selected cell.

<img src="img/image73.png" alt="Figure"/> Select or unselect cells from ID to ID.

<img src="img/image74.png" alt="Figure"/>

<img src="img/image75.png" alt="Figure"/> Select or unselect cells with a string in their name.

Pressing the CTRL key allows you to select or unselect cells manually when clicked on them.

<img src="img/image76.png" alt="Figure"/> This also updates the name and color of the selected cells.

<img src="img/image77.png" alt="Figure"/> This allows the user to select a group of cells automatically.

<img src="img/image78.png" alt="Figure"/> Select cells that are closer than the specified distance:

<img src="img/image79.png" alt="Figure"/>

<img src="img/image80.png" alt="Figure"/> Select cells that are within a cube of the specified width:

<img src="img/image81.png" alt="Figure"/>

<img src="img/image82.png" alt="Figure"/> Select n cells around the selected cell.

<img src="img/image83.png" alt="Figure"/>

<img src="img/image84.png" alt="Figure"/> if 1 select neighbor cell (cell that receive projection from selected cell) if 2 select neighbor of neighbor. if 3, select neighbor of neighbor of neighbor….

<img src="img/image85.png" alt="Figure"/> Select n cells randomly.

<img src="img/image86.png" alt="Figure"/> This will apply the current variable list in the left panel to the selected cells.

# Launch the simuation

<img src="img/image87.png" alt="Figure"/> This will reset all variable states of the model to zero.

<img src="img/image88.png" alt="Figure"/> Run the simulation. A window will be displayed. 

<img src="img/image89.png" alt="Figure"/>

Once the simulation has finished, the results will be plotted in the center of the software.

# Compute the LFP

<img src="img/image90.png" alt="Figure" width="400"/>

<img src="img/image91.png" alt="Figure" width="300"/>set the coordinate for the electrode.

<img src="img/image92.png" alt="Figure" width="70"/> Compute the LFP.

The LFP can be computed with two types of theories. The CSD and the Dipole

<img src="img/image93.png" alt="Figure" width="70"/>

The electrode can be selected as three diferent shape
<img src="img/image94.png" alt="Figure" width="70"/>

Point: 
<img src="img/image95.png" alt="Figure" width="100"/>

Disk:
<img src="img/image96.png" alt="Figure" width="100"/>

Cylinder:
<img src="img/image97.png" alt="Figure" width="100"/>


<img src="img/image98.png" alt="Figure" width="300"/> This is used to control the radius the tilts (for the disk and cylinder) and the height (for the cylinder)

Additionally, a butterworth filter can be processed on the LFP the active button here is checked:


<img src="img/image99.png" alt="Figure" width="250"/>
