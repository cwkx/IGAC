#include "cg_headers.hpp"

void Compute::setup(cl_device_type type, const string &targetVendor, int deviceID)
{
	// Get platform
	try {
		platform = cl::Platform::getDefault();

		if (targetVendor != "default")
		{
			vector<cl::Platform> platforms;
			cl::Platform::get(&platforms);
			vector<cl::Platform>::const_iterator plitr;
			for (plitr = platforms.begin(); plitr!=platforms.end(); ++plitr)
			{
				string platformVendor;
				plitr->getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
				transform(platformVendor.begin(), platformVendor.end(), platformVendor.begin(), ::tolower);

				if (platformVendor.find(targetVendor) != string::npos) {
					platform = *plitr;
					break;
				}
			}
		}
	} catch (cl::Error &e) {
		cout << "Could not set the platform: " << e.what() << endl;
	}

	// Get devices
	try {
		platform.getDevices(type, &devices);
	} catch (cl::Error &e)
	{
		cout << "Could not get devices for specified type: " << e.what() << endl;
	}

#ifdef _WIN32
		cl_context_properties cps[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
			CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
			CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
			0
		};
#elif defined(__linux__)
		cl_context_properties cps[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
			CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
			CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
			0
		};
#elif defined(__APPLE__)
		CGLContextObj glContext = CGLGetCurrentContext();
		CGLShareGroupObj shareGroup = CGLGetShareGroup(glContext);
		cl_context_properties cps[] = {
			CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
			(cl_context_properties)shareGroup,
		};
#endif

	try {
		context = cl::Context(devices, cps);
		selectedDevice = deviceID < 0 ? maxFlopsDevice() : deviceID;
		queue = cl::CommandQueue(context, devices[selectedDevice], CL_QUEUE_PROFILING_ENABLE);
	}
	catch (cl::Error &err)
	{
		cout << "Could not create context, devices, or queue: " << err.what() << " (error: " << err.err() << ")\n";
		cout << "Do you have the NVIDIA, AMD, or INTEL graphics/openCL driver installed?" << endl;
		exit(1);
	}

	cout << "OpenCL/GL Context"
		<< "\nName: " << devices[selectedDevice].getInfo<CL_DEVICE_NAME>()
		<< "\nVendor: " << devices[selectedDevice].getInfo<CL_DEVICE_VENDOR>()
		<< "\nDriver Version: " << devices[selectedDevice].getInfo<CL_DRIVER_VERSION>()
		<< "\nDevice Profile: " << devices[selectedDevice].getInfo<CL_DEVICE_PROFILE>()
		<< "\nDevice Version: " << devices[selectedDevice].getInfo<CL_DEVICE_VERSION>()
		<< "\nMax Work Group Size: " << devices[selectedDevice].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
		<< endl;
}

int Compute::maxFlopsDevice()
{
	int maxFLOPSDevice = 0;
	int maxflops = -1;

	for (unsigned int i=0; i<devices.size(); i++) {

		cl::Device device = devices[i];

		cl_uint maxComputeUnits   = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		cl_uint maxClockFrequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

		int flops = maxComputeUnits*maxClockFrequency;
		cout << "Device: " << i << " has " << flops << " units*freq" << endl;

		if(flops > maxflops) {
			maxflops = flops;
			maxFLOPSDevice = i;
		}
	}

	cout << "Choosing fastest device: " << maxFLOPSDevice << endl;

	return maxFLOPSDevice;
}

string readFile(const string &filename)
{
	ifstream t(filename.c_str());
	return string((istreambuf_iterator<char>(t)),
				   istreambuf_iterator<char>());
}

void Program::build(Compute &compute, const string &filename)
{
	const string src = readFile(filename);
	try
	{
		cl::Program::Sources source(1, make_pair(src.c_str(), src.size()+1));
		data = cl::Program(compute.context, source);
	}
	catch (cl::Error &err) {
		cout << "Program create error: " << err.what() << "\n";
	}
	try
	{
		data.build(compute.devices);
	}
	catch (cl::Error &err) {
		cout << "Error: " << filename << " (build failed at " << err.what() << ")\n";
		cout << data.getBuildInfo<CL_PROGRAM_BUILD_LOG>(compute.devices[compute.selectedDevice]) << endl;
		exit(1);
	}
}
