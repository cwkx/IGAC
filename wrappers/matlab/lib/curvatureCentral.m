function K = curvatureCentral(phi) % div(grad phi / |grad phi|)
    [bdx,bdy]=gradient(phi);
    mag_bg=sqrt(bdx.^2+bdy.^2)+1e-10;
    nx=bdx./mag_bg;
    ny=bdy./mag_bg;
    [nxx,~]=gradient(nx);
    [~,nyy]=gradient(ny);
    K=nxx+nyy;
end