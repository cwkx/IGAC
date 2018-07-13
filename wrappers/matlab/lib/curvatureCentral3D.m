function K = curvatureCentral(phi) % div(grad phi / |grad phi|)
    [bdx,bdy,bdz]=gradient(phi);
    mag_bg=sqrt(bdx.^2+bdy.^2+bdz.^2)+1e-10;
    nx=bdx./mag_bg;
    ny=bdy./mag_bg;
    nz=bdz./mag_bg;
    [nxx,nxy,nxz]=gradient(nx);
    [nyx,nyy,nyz]=gradient(ny);
    [nzx,nzy,nzz]=gradient(nz);
    K=nxx+nyy+nzz;
end