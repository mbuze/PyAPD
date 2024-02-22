%%% MTEX script for generating an IPF of our optimal APD

clear

startup_mtex

%%
% we may want to adjust fonts etc...
setMTEXpref('FontSize',16);
setMTEXpref('xAxisDirection','north');
setMTEXpref('zAxisDirection','outOfPlane');


%%
% crystal symmetry
CS = {... 
  'notIndexed',...
  crystalSymmetry('cubic', [2.9 2.9 2.9], 'mineral', 'Iron (Alpha)', 'color', [0.53 0.81 0.98])};
 

%%

% We load the mock .ang file we prepared based on our optimal APD
fileName = [mtexDataPath filesep 'EBSD' filesep 'ebsd_data_for_MTEX.ang'];
ebsd = EBSD.load(fileName,CS,'interface','ang','convertSpatial2EulerReferenceFrame','setting 2')

%%

% We ask MTEX to find grains:
[grains,ebsd.grainId,ebsd.mis2mean] = calcGrains(ebsd('indexed'),'angle',2*degree);
ebsd(grains(grains.grainSize<=10)) = [];
[grains,ebsd.grainId,ebsd.mis2mean] = calcGrains(ebsd('indexed'),'angle',2*degree);


%%

% and we plot them!

h = figure;
plot(ebsd('Iron (Alpha)'),ebsd('Iron (Alpha)').orientations,'micronbar','on')

% start overide mode
hold on

plot(grains.boundary,'linewidth',2)