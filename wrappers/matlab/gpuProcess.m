function out = gpuProcess(im, gpuParams)
%% This wrapper function creates a temporary .tif image and parses commandline 
% parameters to our executable, then reads the result back into MATLAB.

	%% Write 2D/3D double matrix as a multipage .tif image
    if exist('tmp.tif', 'file') == 2
        delete('tmp.tif');
    end
    if nargin>1
        if strcmp(gpuParams{end,1},'--phi')
            if exist('phi.tif', 'file') == 2
                delete('phi.tif');
            end
        end
    end
    
    
    %% Make sure image being written is normalized
    im = (im - min(im(:)))/(max(im(:))-min(im(:)));
    
    for K=1:length(im(1, 1, :))
       imwrite(im(:, :, K), 'tmp.tif', 'WriteMode', 'append', 'Compression', 'none');
    end
    
    %% Add phi image
    if nargin>1
        if strcmp(gpuParams{end,1},'--phi')
            for K=1:size(im,3)
                phi = gpuParams{end,2};
                imwrite(phi(:, :, K), 'phi.tif', 'WriteMode', 'append', 'Compression', 'none');
            end
        end
    end
    
    %% Get params string
    if nargin>1
        if strcmp(gpuParams{end,1},'--phi')
            params = '-i tmp.tif -o tmp.tif --phi phi.tif';
            gpuParams(end,:) = [];
        else
            params = '-i tmp.tif -o tmp.tif';
        end
        for K=1:length(gpuParams)
            params = strcat(params, [' ', num2str(gpuParams{K,1})], [' ', num2str(gpuParams{K,2})]);
        end
    else
        params = '-i tmp.tif -o tmp.tif';
    end
    
    %% Launch gpu .exe .bin dependening on operating system
    setenv('LD_LIBRARY_PATH')
    system(sprintf('./gpu-active-contour -p nvidia %s', params));
    
    %% Read the result back into matlab
    out = open3D('tmp.tif');
    disp('MATLAB read GPU output successfully!');
    delete('tmp.tif');

end
