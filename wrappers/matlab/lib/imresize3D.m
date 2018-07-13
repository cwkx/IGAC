function out = imresize3D(im, scale)
    im = (im - min(im(:)))/(max(im(:))-min(im(:)));
    t = affine3d([scale(1) 0 0 0; 0 scale(2) 0 0; 0 0 scale(3) 0; 0 0 0 1;]);
    out = imwarp(im,  t);
end