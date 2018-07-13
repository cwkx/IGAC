function drawPart(model, im, scale, slice, isolevel, color, alpha, lightAngle, colmap, meta)

    %% Plot 3D zero-isosurface and output 3D printing .stl file
    t = affine3d([scale(1) 0 0 0; 0 scale(2) 0 0; 0 0 scale(3) 0; 0 0 0 1;]);
    s = imwarp(model, t);
    i = imwarp(im,  t);
    s = padarray(s, [1,1,1], 1e10); % pad to close the isosurface on boundary
    i = padarray(i, [1,1,1], 0);
    
    if (meta)
        figure, imshow(i(:,:, slice)), colormap(colmap),
    end
    
    patch(isosurface(s, isolevel), ...
          'FaceColor', color, 'FaceAlpha', alpha, ...
          'EdgeColor', 'none');  axis on, grid on,

    if (meta)
        camlight, view(230,30), axis('vis3d'), lightangle(lightAngle(1),lightAngle(2)), lighting gouraud;
        xlabel X, ylabel Y, zlabel Z, grid on, box on, ax = gca; ax.BoxStyle = 'full';
        set(gcf, 'Color', [1,1,1]);
    end
    
    drawnow;

end


% %% Plot 3D zero-isosurface and output 3D printing .stl file
% drawpart(fleshCut,   im, [1 1 1], 120, 0, [.75 .55 .45], 1,   230, true);
% drawpart(jaw,        im, [1 1 1], 120, 0, [.65 .45 .45], 0.5, 230, false);
% drawpart(frontTeeth, im, [1 1 1], 120, 0, [.75 .75 .75], 1,   230, false);      
% drawpart(midTeeth,   im, [1 1 1], 120, 0, [.75 .75 .75], 1,   230, false);   
% drawpart(backTeeth,  im, [1 1 1], 120, 0, [.75 .75 .75], 1,   230, false);  

% scale = [1 1 1];
% slice = 28;%floor(size(im,3)/2);
% lightAngle = [-45, 30];
% drawpart(skullCut,   im, scale, slice, -2.2, [.75 .55 .45], 1, lightAngle, 'pink', true);
% drawpart(brainCut,   im, scale, slice, 0,  [.45 .65 .75], 1, lightAngle, 'pink', false);
% drawpart(fluid,      im, scale, slice, 0,  [.75 .75 .65], 1, lightAngle, 'pink', false);
% drawpart(rightEye,   im, scale, slice, 0,  [.75 .75 .75], 1, lightAngle, 'pink', false);
% drawpart(eyeVessels, im, scale, slice, 0,  [.75 .45 .45], 1, lightAngle, 'pink', false);

