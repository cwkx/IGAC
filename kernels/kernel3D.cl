#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
constant sampler_t renderSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
constant sampler_t renderNSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/**
 *  NOTE! In this source the uses of buffers A,B,C have no respect to the actual linking order!
 *  	  A,B,C here are simply temporary variable names for any buffer.
 */

// Checkerboard: float s = (coord.x%2 == coord.y%2 != coord.z%2);

#define PI 3.14159265f
#define TPI 0.63661977f
#define RPI 0.318309886f
#define LOGSQRTAU 0.9189385332f
#define RRPP 1

/** pre-process: pre-compute the KI and KI2H terms **/
kernel void prepare(read_only image3d_t A,
					write_only image3d_t B)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float4 a = read_imagef(A, imageSampler, coord);

	write_imagef(B, coord, (float4)(a.w,
									a.w*a.w,
									0,
									0));
}

/** simply do a horizontal gaussian pass on a 4-channel input image **/
kernel void horzGausA(read_only image3d_t A,
					  write_only image3d_t B,
					  constant float *kSigma,
					  const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x+i, coord.y, coord.z, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, sum);
}

/** simply do a vertical gaussian pass on a 4-channel input image **/
kernel void vertGausA(read_only image3d_t A,
					  write_only image3d_t B,
					  constant float *kSigma,
					  const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x, coord.y+i, coord.z, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, sum);
}

/** simply do a depth gaussian pass on a 4-channel input image **/
kernel void depthGausA(read_only image3d_t A,
					   write_only image3d_t B,
					   constant float *kSigma,
					   const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x, coord.y, coord.z+i, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, sum);
}

/** pre-process: compose im so its (xyzw) = (KI,KI2,phi,im) **/
kernel void compose(read_only image3d_t A,
					read_only image3d_t B,
					write_only image3d_t C,
					const float cx, const float cy, const float cz, const float cr, const int state)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

	const float4 a = read_imagef(A, imageSampler, coord);
	const float4 b = read_imagef(B, imageSampler, coord);

	float phi = length((float4)(coord.x-cx, coord.y-cy, coord.z-cz, 0)) - cr;

	if (state == 0)
	{
		// None
		phi = sign(phi)*2.0f;
	}
	else if (state == 1)
	{
		// Add
		if (phi > 0)
			phi = b.z;
		else
			phi = min(phi, b.z);
	}
	else if (state == 2)
	{
		// Neutral
		if (phi > 0)
			phi = b.z;
		else
			phi = 2.0;
	}
	else if (state == 3)
	{
		// Barrier
		if (phi > 2)
			phi = b.z;
		else
			phi = 100000000;
	}

	write_imagef(C, coord, (float4)(b.x, b.y, phi, a.w));
}

/** copy from select channel with with neumann boundary condition on phi **/
kernel void neumannCopyA(read_only image3d_t A,
					read_only image3d_t B,
				    write_only image3d_t C, const int readPhi, const int width, const int height, const int depth)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float4 a = read_imagef(A, imageSampler, coord);
	const float4 b = read_imagef(B, imageSampler, coord);

	const int4 neumannOffset = (int4)(coord.x + 2 * (coord.x==0) - 2 * (coord.x==(width-1)),
									  coord.y + 2 * (coord.y==0) - 2 * (coord.y==(height-1)),
									  coord.z + 2 * (coord.z==0) - 2 * (coord.z==(depth-1)),
									  0);
	if (readPhi == 0)
		write_imagef(C, coord, (float4)(a.x, a.y, read_imagef(A, imageSampler, neumannOffset).z, a.w));
	else
		write_imagef(C, coord, (float4)(a.x, a.y, read_imagef(B, imageSampler, neumannOffset).x, a.w));
}

/** compute normalised grad (xy), and neumann boundary condition for phi (z) **/
kernel void normalisedGradPhi(read_only image3d_t A,
						   	  write_only image3d_t B,
							  int width, int height, int depth)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

	// gradient using [-0.5 , 0, +0.5]
	float3 g = (float3)(read_imagef(A, imageSampler, (int4)(coord.x+1,coord.y, coord.z, 0)).z -
					    read_imagef(A, imageSampler, (int4)(coord.x-1,coord.y, coord.z, 0)).z,
						read_imagef(A, imageSampler, (int4)(coord.x,coord.y+1, coord.z, 0)).z -
						read_imagef(A, imageSampler, (int4)(coord.x,coord.y-1, coord.z, 0)).z,
						read_imagef(A, imageSampler, (int4)(coord.x,coord.y, coord.z+1, 0)).z -
						read_imagef(A, imageSampler, (int4)(coord.x,coord.y, coord.z-1, 0)).z) * 0.5f;

	// normalised grad
	const float gm = sqrt(g.x*g.x + g.y*g.y + g.z*g.z)+1e-10f; // precision warning
	g /= gm;
	const float phi = read_imagef(A, imageSampler, coord).z;

	write_imagef(B, coord, (float4)(g.x,g.y,g.z,phi));
}

/** compute inner terms for KIH, KH, and KI2H **/
// input  A = KI,  KI2, ~,    im
// input  B = gx,  gy,  gz,   phi
// output C = KIH, KH,  KI2H, div
kernel void divPrepFirstFilter(read_only image3d_t A,
							   read_only image3d_t B,
					   	   	   write_only image3d_t C)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

	const float div = (read_imagef(B, imageSampler, (int4)(coord.x+1,coord.y,coord.z,0)).x - // nxx
				  	   read_imagef(B, imageSampler, (int4)(coord.x-1,coord.y,coord.z,0)).x +
					   read_imagef(B, imageSampler, (int4)(coord.x,coord.y+1,coord.z,0)).y - // nyy
					   read_imagef(B, imageSampler, (int4)(coord.x,coord.y-1,coord.z,0)).y +
					   read_imagef(B, imageSampler, (int4)(coord.x,coord.y,coord.z+1,0)).z - // nzz
					   read_imagef(B, imageSampler, (int4)(coord.x,coord.y,coord.z-1,0)).z) * 0.5f;

	// get heaviside, dirac, and phi
	const float phi = read_imagef(B, imageSampler, coord).w;
	const float heav = 0.5f*(1.0f + TPI * atan(phi));
	const float im  = read_imagef(A, imageSampler, coord).w;

	// return KIH, KH, and KI2H inner terms
	write_imagef(C, coord, (float4)(heav*im,
									heav,
									im*im*heav,
									div));
}

/** horizontal gaussian pass on a 4-channel input image, passthrough w component **/
kernel void horzGausB(read_only image3d_t A,
					  write_only image3d_t B,
					  constant float *kSigma,
					  const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float div = read_imagef(A, imageSampler, coord).w;

	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x+i, coord.y, coord.z, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, (float4)(sum.x,sum.y,sum.z,div));
}

/** vertical gaussian pass on a 4-channel input image, passthrough w component **/
kernel void vertGausB(read_only image3d_t A,
					  write_only image3d_t B,
					  constant float *kSigma,
					  const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float div = read_imagef(A, imageSampler, coord).w;

	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x, coord.y+i, coord.z, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, (float4)(sum.x,sum.y,sum.z,div));
}

/** depth gaussian pass on a 4-channel input image, passthrough w component **/
kernel void depthGausB(read_only image3d_t A,
					   write_only image3d_t B,
					   constant float *kSigma,
					   const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float div = read_imagef(A, imageSampler, coord).w;

	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x, coord.y, coord.z+i, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, (float4)(sum.x,sum.y,sum.z,div));
}

/** contains most of the body of the work, passthrough div **/
kernel void prepSecondFilter(read_only image3d_t A,
							 read_only image3d_t B,
				 	 	     write_only image3d_t C,
							 const float2 lambda)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

	const float4 a = read_imagef(A, imageSampler, coord); // KI, KI2, ~,    im
	const float4 b = read_imagef(B, imageSampler, coord); // KIH, KH, KI2H, div

	const float KI   = a.x; // gauss(im)
	const float KI2  = a.y; // gauss(im^2)
	const float KIH  = b.x;
	const float KH   = b.y;
	const float KI2H = b.z;

	const float u1 = KIH/KH;
	const float u2 = (KI - KIH)/(1.0f - KH);

	float sigma1 = (KI2H / KH) - u1*u1;
	float sigma2 = ((KI2-KI2H)/(1.0f-KH))-u2*u2;

	const float Ax = lambda.x*log(sqrt(sigma1)) - lambda.y*log(sqrt(sigma2)) + lambda.x*u1*u1 /(2.0f*sigma1) - lambda.y*u2*u2/(2.0f*sigma2);
	const float Ay = lambda.y*u2/sigma2 - lambda.x*u1/sigma1;
	const float Az = lambda.x*1.0f/(2.0f*sigma1) - lambda.y*1.0f/(2.0f*sigma2);

	write_imagef(C, coord, (float4)(Ax, Ay, Az, b.w));
}

/** horizontal gaussian pass on a 4-channel input image, passthrough w component **/
kernel void horzGausC(read_only image3d_t A,
					  write_only image3d_t B,
					  constant float *kSigma,
					  const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float div = read_imagef(A, imageSampler, coord).w;

	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x+i, coord.y, coord.z, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, (float4)(sum.x,sum.y,sum.z,div));
}

/** vertical gaussian pass on a 4-channel input image, passthrough w component **/
kernel void vertGausC(read_only image3d_t A,
					  write_only image3d_t B,
					  constant float *kSigma,
					  const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float div = read_imagef(A, imageSampler, coord).w;

	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x, coord.y+i, coord.z, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, (float4)(sum.x,sum.y,sum.z,div));
}

/** depth gaussian pass on a 4-channel input image, passthrough w component **/
kernel void depthGausC(read_only image3d_t A,
					   write_only image3d_t B,
					   constant float *kSigma,
					   const int kSize)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float div = read_imagef(A, imageSampler, coord).w;

	const int start = -kSize/2;
	const int end = -start;
	float4 sum = (float4)(0.0f);

	for(int i=start; i<=end; ++i)
	{
		const int4 offset = (int4)(coord.x, coord.y, coord.z+i, 0);
		sum += read_imagef(A, imageSampler, offset) * kSigma[i-start];
	}

	write_imagef(B, coord, (float4)(sum.x,sum.y,sum.z,div));
}

kernel void updatePhi(read_only  image3d_t A,
					  read_only  image3d_t B,
					  write_only image3d_t C,
					  int width, int height, int depth,
					  const float2 lambda,
					  const float4 params)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float4 a = read_imagef(A, imageSampler, coord); // KI, KI2, phi, im
	const float4 b = read_imagef(B, imageSampler, coord); // Ax, Ay, Az, div

	float localForce = (lambda.x - lambda.y) * LOGSQRTAU
					 + b.x
					 + a.w*b.y
					 + a.w*a.w * b.z;

	if (isnan(localForce)) localForce = 0.0f;

	const float phi = read_imagef(A, imageSampler, coord).z;
	const float dirac = RPI/(1.0f+phi*phi)+1e-10f;
	const float curv = b.w;

	const float p = read_imagef(A, imageSampler, (int4)(coord.x-1,coord.y,coord.z, 0)).z +
			  	    read_imagef(A, imageSampler, (int4)(coord.x+1,coord.y,coord.z, 0)).z +
					read_imagef(A, imageSampler, (int4)(coord.x,coord.y+1,coord.z, 0)).z +
					read_imagef(A, imageSampler, (int4)(coord.x,coord.y-1,coord.z, 0)).z +
					read_imagef(A, imageSampler, (int4)(coord.x,coord.y,coord.z+1, 0)).z +
					read_imagef(A, imageSampler, (int4)(coord.x,coord.y,coord.z-1, 0)).z +
					-6.0f * phi;

	const float aF = -params.w * dirac * localForce;
	const float pF = params.y * (p-curv);
	const float lF = params.z*dirac*curv;
	float newPhi = phi + params.x*(lF+pF+aF);

	if (newPhi > 1000000)
		newPhi = phi;

	write_imagef(C, coord, (float4)(a.x,a.y,newPhi,a.w));
}

/** copy with neumann boundary condition on phi **/
kernel void neumannCopyB(read_only image3d_t A,
				    	 write_only image3d_t B, const int width, const int height, const int depth)
{
	const int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	const float4 a = read_imagef(A, imageSampler, coord);

	const int4 neumannOffset = (int4)(coord.x + 2 * (coord.x==0) - 2 * (coord.x==(width-1)),
									  coord.y + 2 * (coord.y==0) - 2 * (coord.y==(height-1)),
									  coord.z + 2 * (coord.z==0) - 2 * (coord.z==(depth-1)),
									  0);

	const float neumannBound = read_imagef(A, imageSampler, neumannOffset).z;

	write_imagef(B, coord, (float4)(a.x, a.y, neumannBound, a.w));
}

inline float3 matVecMul(float16 mat, float4 vec)
{
	return (float3)(dot(mat.s048C, vec),dot(mat.s159D, vec),dot(mat.s26AE, vec));
}

inline float noBarrier(float x)
{
	return x > 1000 ? 0.01 : x;
}

inline float distPhi(read_only image3d_t A, uint maxDim, float4 pos)
{
	return noBarrier(read_imagef(A, renderSampler, pos).z / maxDim);
}

float calcAO(read_only image3d_t A, uint maxDim, float4 pos, float4 nor)
{
	float occ = 0.0;
	float sca = 1.0;
	for( int i=0; i<50; i++ )
	{
		float hr = 0.01 + 0.12*(float)(i*0.1f)/4.0;
        float4 aopos =  nor * hr + pos;
        float dd = distPhi(A, maxDim, nor * hr + pos);
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
	return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );
}

float softShadow(read_only image3d_t A, uint maxDim, float4 ro, float4 rd, float mint, float tmax )
{
	float res = 1.0f;
    float t = mint;
    for( int i=0; i<60; i++ )
    {
		float h = distPhi(A, maxDim, ro + rd*t);
        res = min( res, 8.0f*h/t );
        t += clamp( h, 0.02f, 0.10f )*0.1;
        if( h<0.001f || t>tmax ) break;
    }
    return clamp( res, 0.0f, 1.0f );
}

inline float maxc(float4 v)
{
	return max(v.x, max(v.y, v.z));
}

inline float4 maxs(float4 v, float s)
{
	return (float4)(max(v.x, s), max(v.y, s), max(v.z, s), 0.0f);
}

/** render to PBO **/
kernel void render(read_only image3d_t A,
				   global uint *B,
				   const int wWidth, const int wHeight,
				   const int iWidth, const int iHeight, const int iDepth,
				   const int zSlice, const float16 mat, const float alpha, const float adelta, const float astretch, const float mx, const float my)
{
	const float2 coord = (float2)(get_global_id(0), get_global_id(1));

	const float4 slice = (float4)(coord.x/wWidth, coord.y/wHeight, (float)(zSlice)/iDepth, 0);
	const float4 a = read_imagef(A, renderSampler, slice);
	const float4 s = read_imagef(A, renderNSampler, slice);
	const float phi = a.z > 1000 ? 0 : a.z;

	// heaviside and dirac
	const float dirac = (RPI/(1.0f+phi*phi)+1e-10f)*1024;
	const float heav = 256-(256*(1.0f + TPI * atan(phi)));

	const float phin = clamp(phi*10,  0.0f, 255.0f);
	const float heavn = clamp(adelta*dirac+(1-adelta)*heav,0.0f,255.0f);
	uint img = 0.7*clamp(s.w*230,0.0f,255.0f);

	uint imgheav = clamp(img+heavn,  0.0f, 255.0f);
	uint imgbarr = (a.z > 1000) ? clamp(img+100.0f, 0.0f, 255.0f) : img;
	uint imgphi = clamp(phin+img,  0.0f, 255.0f);

	float cursor = sqrt((coord.x-mx)*(coord.x-mx) + (coord.y-my)*(coord.y-my)) - 5;
		  cursor = (1.0f-alpha)*(1.0f-(1.0f+tanh(2.0f*cursor))*0.5f);

	imgheav = mix((float)(imgheav),255.0f, cursor);
	imgbarr = mix((float)(imgbarr),255.0f, cursor);
	img = mix((float)(img),255.0f, cursor);

	// basic 2D slice view
	if (alpha <= 0.0f)
	{
		//            		A            B               G             	   R
		const uint color = 255 | (imgbarr << 8) | (img << 16) | (imgheav << 24);
		B[(int)(coord.y * wWidth + coord.x)] = color;
		return;
	}
	// ray marcher
	else
	{
		// lights and setup
		float t = 0;
		float4 pos;
		float3 n = (float3)(0,0,0);
		uint4 cold = (uint4)(255,255,255,1);
		float3 csum = (float3)(0,0,0);
		float bval = 0.0;

		for (int rrpp=0; rrpp<(RRPP*RRPP); rrpp++)
		{
			int rx = rrpp % RRPP;
			int ry = rrpp / RRPP;
			float d = 1.0f / RRPP;
			float px = 2.0f / wWidth;
			float py = 2.0f / wHeight;

			const uint maxDim = max(iWidth, max(iHeight, iDepth));
			const float marchingStep = 0.3;
			const int steps = 8192;

			const float3 roPerspective = (float3)mat.sCDE;						// camera position
			const float4 rayDir0Perspective = (float4)(px*d*rx + 2*slice.x-1, py*d*ry + 2*slice.y-1, 1, 0);	// in camera's frame of reference
			const float3 rdPerspective = normalize(matVecMul(mat, rayDir0Perspective));				// in world's frame of reference

			const float4 roOrtho0 = (float4)(px*d*rx + 2*slice.x-1, py*d*ry + 2*slice.y-1, 0, 0);
			const float3 roOrtho = matVecMul(mat, roOrtho0) + roPerspective;
			const float3 rdOrtho = mat.s89A;

			// lerp between perspective and ortho modes:
			const float3 ro = mat.sF*roPerspective + (1-mat.sF)*roOrtho;
			const float3 rd = mat.sF*rdPerspective + (1-mat.sF)*rdOrtho;

			const float3 boxScale = mix((float3)(1,1,1), (float3)(iWidth,iHeight,iDepth) / maxDim, astretch);
			const float3 bmin = (float3)(-0.5,-0.5,-0.5) * boxScale;
			const float3 bmax = (float3)( 0.5, 0.5, 0.5) * boxScale;

			// analytical ray hit box
			const float3 omin = (bmin - ro ) / rd; // inner term is bmin
			const float3 omax = (bmax - ro ) / rd; // outter term is bmax
			const float3 kmax = max ( omax, omin );
			const float3 kmin = min ( omax, omin );
			const float  tmax = min ( kmax.x, min ( kmax.y, kmax.z ) );
			const float  tmin = max ( max ( kmin.x, 0.0f ), max ( kmin.y, kmin.z ) );

			bval = tmax-tmin;


			// ray hit box
			if (tmax > tmin)
			{
				t = tmin;
				const float4 boxScale4 = (float4)(boxScale, 0);

				for (int i=0; i<steps; ++i)
				{
					pos = (float4)((ro.x+rd.x*t), (ro.y+rd.y*t), (ro.z+rd.z*t), 0.0);
					pos = (pos+boxScale4/2) / boxScale4;

					if (t<0.0001 || t>tmax )
						break;

					t+=distPhi(A,maxDim,pos)*marchingStep;
				}

				if (t<tmax)
				{
					const float h = 0.01f; // central difference
					pos = (float4)((ro.x+rd.x*t), (ro.y+rd.y*t), (ro.z+rd.z*t), 1.0);
					pos = (pos+boxScale4/2) / boxScale4;

					const float4 eps = (float4)(h, 0, 0, 0);

					// compute normal
					n = normalize((float3)(distPhi(A, maxDim, pos+eps.xyzw)-distPhi(A, maxDim, pos-eps.xyzw),
										   distPhi(A, maxDim, pos+eps.yxzw)-distPhi(A, maxDim, pos-eps.yxzw),
										   distPhi(A, maxDim, pos+eps.zyxw)-distPhi(A, maxDim, pos-eps.zyxw)));

					t = 0.3+0.7*clamp(1.5f*read_imagef(A, renderSampler, pos).w, 0.0f, 1.0f);

					// lighting
					const float3 ref = rd - 2.0f * dot(n, rd) * n;
					const float occ = calcAO(A, maxDim, pos, (float4)(n.x,n.y,n.z,1));
					const float3 lig = normalize((float3)(-0.6f, 0.7f, -0.5f) );//;(float3)(-0.6f, 0.7f, -0.5f) );
					const float amb = clamp( 0.5f+0.5f*n.y, 0.0f, 1.0f );

					float dif = clamp( dot( n, lig ), 0.0f, 1.0f );
					float difn = clamp( dot( n, -lig ), 0.0f, 1.0f );
					const float bac = clamp( dot( n, normalize((float3)(-lig.x,0.0,-lig.z))), 0.0f, 1.0f )*clamp( 1.0f-pos.y,0.0f,1.0f);
					float dom = smoothstep( -0.1f, 0.1f, ref.y );
					const float fre = pow(clamp(1.0f+dot(n,rd),0.0f, 1.0f), 2.0f );
					const float spe = pow(clamp( dot( ref, lig ), 0.0f, 1.0f ),16.0f);

					dif  *= softShadow(A, maxDim, pos, (float4)(lig.x,lig.y,lig.z,1.0f), 0.02f, 2.5f );
					dom  *= softShadow(A, maxDim, pos, (float4)(ref.x,ref.y,ref.z,1.0f), 0.02f, 2.5f );

					// material properties
					float3 lin = (float3)(0.0f);
					lin += 1.80f*dif*(float3)(0.45,0.65,0.75);
					lin += 1.80f*difn*(float3)(0.75,0.55,0.55);
					lin += 1.20f*spe*(float3)(0.75,0.45,0.45)*dif;
					lin += 1.00f*amb*(float3)(0.75,0.45,0.45)*occ; lin = clamp(lin, 0.0f, 1.0f);
					lin += 0.30f*dom*(float3)(0.50,0.70,1.00)*occ;
					lin += 0.30f*bac*(float3)(0.25,0.25,0.25)*occ;
					lin += 0.40f*fre*(float3)(1.00,1.00,1.00)*occ;
					float3 c = (float3)(t,t,t)*(lin);
					csum += mix(c, (float3)(0.8f,0.9f,1.0f), 1.0f-exp( -0.002f*t*t ) );
				}
				else
				{
					csum += (float3)(1.0f,1.0f,1.0f);
				}
			}
			else
			{
				csum += (float3)(1.0f,1.0f,1.0f);
			}
		}

		csum = csum / (RRPP*RRPP);
		cold = (uint4)(clamp(csum.x*255, 0.0f, 255.0f),clamp(csum.y*255, 0.0f, 255.0f),clamp(csum.z*255, 0.0f, 255.0f), 0);

		// badly written color lerp
		uint boxd = clamp(255-bval*30, 0.0f, 255.0f);
		uint4 limg = (uint4)(imgheav, img, imgbarr, 0);

		if (cold.x == 255) cold.x = boxd;
		if (cold.y == 255) cold.y = boxd;
		if (cold.z == 255) cold.z = boxd;

		cold = (uint4)(clamp(cold.x*alpha, 0.0f, 255.0f), clamp(cold.y*alpha, 0.0f, 255.0f), clamp(cold.z*alpha, 0.0f, 255.0f), 0.0f);
		limg = (uint4)(clamp(limg.x*(1.0f-alpha), 0.0f, 255.0f), clamp(limg.y*(1.0f-alpha), 0.0f, 255.0f), clamp(limg.z*(1.0f-alpha), 0.0f, 255.0f), 0.0f);
		cold += limg;

		//            		 A          B                G                 R
		const uint color = 255 | (cold.z << 8) | (cold.y << 16) | (cold.x << 24);
		B[(int)(coord.y * wWidth + coord.x)] = color;
	}
}
