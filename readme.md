## Paper

This code implements "Interactive GPU active contours for segmenting inhomogeneous objects". Please cite if you find it useful.
https://link.springer.com/article/10.1007/s11554-017-0740-1

## Common Mistakes

- Make sure kernels directory is next to the compiled binary
- If on linux, make sure "glxgears" works
- Make sure your drivers are up to date

## Readme

- Compile using CMake
- Typical usage scenario:

```
./IGAC -p nvidia -i data/noiseTumour3D.tif
```

## Hotkeys

[Click here to view application hotkeys](https://github.com/cwkx/IGAC/blob/master/readme.pdf)

## Command Line Options

```
USAGE:

   ./IGAC-arch  -i <string> -p <string> [-o <string>] [--phi <string>] [-d
                <string>] [--device <int>] [-r <float>] [-s <float>] [-g
                <float>] [--maxiter <float>] [--cr <float>] [--cz <float>]
                [--cy <float>] [--cx <float>] [--awgn <float>] [--lambda2
                <float>] [--lambda1 <float>] [--alf <float>] [--nu <float>]
                [--mu <float>] [--timestep <float>] [--sigma <float>] [--]
                [--version] [-h]

Where:

   -i <string>,  --image <string>
     (required)  Tiff image [path/image.tif]

   -p <string>,  --platform <string>
     (required)  Platform ID string [intel, nvidia, default, ...]

   -o <string>,  --output <string>
     Optional Save Tiff [path/filename.tif]

   --phi <string>
     Optional Tiff image for initial phi seed region [path/phi.tif]

   -d <string>,  --type <string>
     Device type [CPU, GPU, ALL, ACCELERATOR, DEFAULT]

   --device <int>
     Choose specific device id rather than max flops [0, 1, ...]

   -r <float>,  --range <float>
     Segmentation range

   -s <float>,  --smooth <float>
     Smoothing weight

   -g <float>,  --grow <float>
     Prefer to shrink or grow

   --maxiter <float>
     (Optional Advanced) Maximum iterations

   --cr <float>
     (Optional Advanced) Seed circle/sphere radius

   --cz <float>
     (Optional Advanced) Seed sphere z position

   --cy <float>
     (Optional Advanced) Seed circle/sphere y position

   --cx <float>
     (Optional Advanced) Seed circle/sphere x position

   --awgn <float>
     (Optional Advanced) AWGN intensity variation to prevent zero division
     (constant)

   --lambda2 <float>
     (Optional Advanced) Function of grow parameter

   --lambda1 <float>
     (Optional Advanced) Function of grow parameter

   --alf <float>
     (Optional Advanced) Data weight term (constant)

   --nu <float>
     (Optional Advanced) Same as curvature term

   --mu <float>
     (Optional Advanced) Signed distance function term (constant)

   --timestep <float>
     (Optional Advanced) Timestep (constant)

   --sigma <float>
     (Optional Advanced) Same as range parameter

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.
```
