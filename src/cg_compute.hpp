struct Compute
{
   cl::Context context;
   cl::CommandQueue queue;
   cl::Platform platform;
   vector<cl::Device> devices;
   int selectedDevice;

   void setup(cl_device_type type, const string &platformVendor, int specificDevice = -1);
   int  maxFlopsDevice();
};

struct Program
{
   cl::Program data;

   void build(Compute &compute, const string &src);
};
