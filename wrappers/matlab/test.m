clear all, close all, clc;

%% Change directory to two up from current script
if(~isdeployed)
  cd(fileparts(which(mfilename)));
  cd '../../'
end

%% Run this script from the top-level directory containing data/...
im = single(open3D('data/vessel2D.tif'));
im = (im - min(im(:)))/(max(im(:))-min(im(:)));
%im = permute(im, [3 1 2]);
%im = imresize3D(im, [1.5 1.5 1.5]);
im = imnoise(im , 'gaussian', 0.0, 0.001); % make sure no divisions by zero


params = {'--platform', 'nvidia';
          '--sigma',     3; % 5.0 for tumour
          '--timestep',  0.1;
          '--mu',        1.0;
          '--nu',        30; % spread  56
          '--alf',       30.0; % da1ta fitting weight 
          '--lambda1',   1.0;
          '--lambda2',   1.05; % bigger to grow 
          '--cx',        floor(size(im,2)/2);   % phi seed x
          '--cy',        floor(size(im,1)/2);
          '--cz',        floor(size(im,3)/2);   % phi seed z (unused in 2D)
          '--cr',        15.0
          '--maxiter'    -1};

cpuPhi = gather(single(cpuProcess(gpuArray(single(im)), params)));

figure, imshow3D(cpuPhi);

