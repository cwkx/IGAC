struct ActiveContour : Renderer
{
   // GPU data
   cl::Buffer K;
   cl::Image3D A;
   cl::Image3D B;
   cl::Image3D C;

   // CPU data
   Image *image;
   Image *phi;
   vector<float> gauss;
   cl_float2 lambda;
   cl_float4 params;
   float sigma;
   float slambda;
   int maxIterations;
   int activeSlice = 0;

   // Profiling data
   vector<vector<float>> timings;

   void setup			(GLFWwindow* window, const Compute &compute, const vector<float> &params);
   void update			(GLFWwindow* window, const Compute &compute, int width, int height);
   void scroll			(double xOffset, double yOffset);
   void shutdown		(const Compute &compute) { save(compute, A, 2); }
   void setGauss		(); 														// set gaussian coefficients based on sigma value
   void mapParam		(int i, float &x, string &name, bool set); 	// get/set parameter data from i'th keyboard key
   void setParams		(GLFWwindow*, const Compute&, double, int, bool&, bool&);
   void save			(const Compute &compute, const cl::Image &im, const int channel=2);
   void logTiming		();
   void printParams		(bool advanced = false);
};
