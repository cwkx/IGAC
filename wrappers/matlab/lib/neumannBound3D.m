function u = neumannBound3D(u) 
[nrow,ncol,nlayer] = size(u);
    coords = (1:numel(u))';
    [I, J, K] = ind2sub(size(u), coords);
    coords = [I,J,K];
    coords(coords(:,1)==1, 1) = coords(coords(:,1)==1, 1) + 2;
    coords(coords(:,1)==nrow, 1) = coords(coords(:,1)==nrow, 1) - 2;
    coords(coords(:,2)==1, 2) = coords(coords(:,2)==1, 2) + 2;
    coords(coords(:,2)==ncol, 2) = coords(coords(:,2)==ncol, 2) - 2;
    coords(coords(:,3)==1, 3) = coords(coords(:,3)==1, 3) + 2;
    coords(coords(:,3)==nlayer, 3) = coords(coords(:,3)==nlayer, 3) - 2;
    coords = sub2ind(size(u), coords(:,1), coords(:,2), coords(:,3));
    u = reshape(u(coords), [nrow, ncol, nlayer]);
return;