%%% MTEX script for generating the grain file

%%

% Loading mtex 
clear
startup_mtex

%%
% For plotting, we may want to start by adjusting fonts etc...
setMTEXpref('FontSize',16);
setMTEXpref('xAxisDirection','north');
setMTEXpref('zAxisDirection','outOfPlane');

%%
% We define the relevant crystal symmetry:
CS = {... 
  'notIndexed',...
  crystalSymmetry('cubic', [2.9 2.9 2.9], 'mineral', 'Iron (Alpha)', 'color', [0.53 0.81 0.98])};

%%
% We load the EBSD file, which we manually loaded into the 'data/EBSD'
% subfolder in MTEX

fileName = [mtexDataPath filesep 'EBSD' filesep 'Sample_LC steel.ang'];
ebsd = EBSD.load(fileName,CS,'interface','ang','convertSpatial2EulerReferenceFrame','setting 2')

% we then gridify it to make computations quicker:
ebsd = ebsd.gridify

%%

% We now ask MTEX to switch from hexagonal grid to square grid
% define a square unit cell
unitCell = [-0.5 -0.5; -0.5 0.5; 0.5 0.5; 0.5 -0.5];
 
% use the square unit cell for gridify
ebsdS = ebsd.gridify('unitCell',unitCell)

%%

% First run of the grain finding algorithm, the criterion is deviations of
% up to 2 degrees in crystallographic orientation:

[grains,ebsdS.grainId,ebsdS.mis2mean] = calcGrains(ebsdS('indexed'),'angle',2*degree);

%%

% As is common practice, we then remove pixels associated with the smallest grains
% we set the threshold to be 10 or less pixels
ebsdS(grains(grains.grainSize<=10)) = [];

% And run the algorithm again -- MTEX is geared towards working with
% "missing" pixels.

[grains,ebsdS.grainId,ebsdS.mis2mean] = calcGrains(ebsdS('indexed'),'angle',2*degree);

% We can see that we obtain a `grains` object with 4587 grains.
%%

% We then create an IPF (this is the plot from the paper)

plotx2east
plotzIntoPlane

%plot(ebsdS,ebsdS.orientations)
h = figure;
plot(ebsdS('Iron (Alpha)'),ebsdS('Iron (Alpha)').orientations,'micronbar','on')

% start overide mode
hold on

plot(grains.boundary,'linewidth',2)

% stop overide mode
hold off

%%

% IPF color key from the paper can be obtained as follows 
ipfKey = ipfColorKey(ebsdS('Iron (Alpha)'));
colors = ipfKey.orientation2color(ebsdS('Iron (Alpha)').orientations);

h = figure;
plot(ipfKey)


%%
% We can now prepare the grain file.
[theta,a,b] = fitEllipse(grains);

areas = grains.area;
X = grains.centroid;
grain_color_map = ipfKey.orientation2color(grains.meanOrientation); % for coloring

T_grain_file = table(X, a, b, theta, areas, grain_color_map)
writetable(T_grain_file, 'sample_lc_steel_grain_file_final.txt','Delimiter',' ')

%%

% To do the pixel-level comparison with the optimal APD, we also need the
% grain map:

T = table(ebsdS('indexed').x, ebsdS('indexed').y, ebsdS('indexed').grainId)
writetable(T, 'grainIDs_final.txt','Delimiter',' ')
%dlmwrite('grainIDs_final.txt',T{:,:},'delimiter',' ','precision','%0.4f %0.4f %0.0f')

%% 

% Finally, to use MTEX to plot our APD-based IPF, we need to know what is
% the mean crystallographic orientation of each grain:
T = table(grains.meanOrientation.phi1, grains.meanOrientation.Phi, grains.meanOrientation.phi2)
writetable(T, 'grains_mean_orientation_final.txt','Delimiter',' ')