function [smoothed] = psmin(a, b, k)
    % polynomial smooth min: http://iquilezles.org/www/articles/smin/smin.htm
    % INPUTS:
    % a - signed distance field a
    % b - signed distance field b
    % k - blending distance
    % OUTPUTS:
    % smoothed - SDFs a and b blended by distance k
    if nargin < 3
        k = 0.1;
    end
    
    h = 0.5+0.5*(b-a)./k;
    h(h<0) = 0;
    h(h>1) = 1;
    smoothed = (1-h).*b + h.*a - k*h.*(1.0-h);
end