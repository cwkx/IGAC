function [D, CC, base] = dist2Mat3DSegments( bw )
    % Computes a squared distance matrix between the centroids of the
    % connected components in the 3D binary image bw
    % PARAMETERS:
    % bw - 3D binary image
    % OUTPUTS:
    % D - squared distance matrix
    % CC - bwconncomp(bw)
    % trunk - struct describing the trunk (x-y position, width, and which
    %         of the rows/cols in D correspond to trunk segments
    
    %% First compute segment centroids (can't use bwmorph, that only does 2D)
    CC = bwconncomp(bw);
    CX = [];
    CY = [];
    CZ = [];
    for i = 1:CC.NumObjects
        [I,J,K] = ind2sub(size(bw), CC.PixelIdxList{i});
        CY(end+1) = mean(I);
        CX(end+1) = mean(J);
        CZ(end+1) = mean(K);
    end
    centroids = [CX; CY; CZ];
    
    %% Now add the base as a grid of blocks
    base = struct();
    base.tileWidth = 21;
    base.height = 1;
    base.shape = [floor(size(bw,1)/base.tileWidth), floor(size(bw,2)/base.tileWidth)];
    base.numTiles = prod(base.shape);
    base.centroids = ceil(base.tileWidth/2) + base.tileWidth*(ceil((1:(base.numTiles))/base.shape(1))-1);                        % x components
    base.centroids = [base.centroids; ceil(base.tileWidth/2) + base.tileWidth*mod(0:(base.numTiles-1),base.shape(1))];           % y components
    base.centroids = [base.centroids; ones(1, base.numTiles)];                                                                   % z components
    
    base.segIdx = (size(centroids,2)+1) : (size(centroids,2)+size(base.centroids,2));
    centroids = cat(2, centroids, base.centroids);
    
    %% Now compute distance matrix
    % we use the method described at https://statinfer.wordpress.com/2011/11/14/efficient-matlab-i-pairwise-distances/
    norm2 = dot(centroids, centroids, 1);
    D = bsxfun(@plus, norm2, norm2');   % here D_ij = norm2(i) + norm2(j)
    D = D - 2 * centroids' * centroids;
    % now D_ij is the squared distance between segment i and segment j
    
end
