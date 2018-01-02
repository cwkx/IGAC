#include "cg_headers.hpp"

// Global data
GLFWwindow* window;
Compute compute;
Image image;
Image phi;
cl_device_type type = CL_DEVICE_TYPE_DEFAULT;
string outputFilename;
string platformVendor;
int specificDevice = -1;
unique_ptr<Renderer> renderer;
vector<float> acParams;

// Callbacks
void windowSizeCallback(GLFWwindow* window, int width, int height)
{
   width = max(width,1);
   height = max(height, 1);
   glLoadIdentity();
   glOrtho(0.0, width, 0.0, height, 0.0, 1.0);
   renderer->resize(compute, width, height);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
   if (renderer)
      renderer->scroll(xoffset, yoffset);
}

// Entry
int main(int argc, char* argv[])
{
   try
   {
      CmdLine cmd("Interactive GPU Active Contours for Segmenting Inhomogeneous Objects.", ' ', "1.2");

      // Advanced parameters (typically defined interactively, as constants, or as a function of the basic parameters)
      ValueArg<float> paramArgSigma		("","sigma",	"(Optional Advanced) Same as range parameter",							false,	3.0f, "float");	cmd.add(paramArgSigma);
      ValueArg<float> paramArgTimestep	("","timestep","(Optional Advanced) Timestep (constant)",								false,	0.1f, "float");	cmd.add(paramArgTimestep);
      ValueArg<float> paramArgMu			("","mu",		"(Optional Advanced) Signed distance function term (constant)",	false,	1.0f, "float");	cmd.add(paramArgMu);
      ValueArg<float> paramArgNu			("","nu",		"(Optional Advanced) Same as curvature term",							false, 50.0f, 	"float");	cmd.add(paramArgNu);
      ValueArg<float> paramArgAlf		("","alf",		"(Optional Advanced) Data weight term (constant)",						false, 30.0f, 	"float");	cmd.add(paramArgAlf);
      ValueArg<float> paramArgLambda1	("","lambda1",	"(Optional Advanced) Function of grow parameter",						false, 1.0f, 	"float");	cmd.add(paramArgLambda1);
      ValueArg<float> paramArgLambda2	("","lambda2",	"(Optional Advanced) Function of grow parameter",						false, 1.05f, 	"float");	cmd.add(paramArgLambda2);
      ValueArg<float> paramArgAWGN		("","awgn",		"(Optional Advanced) AWGN intensity variation to prevent zero division (constant)", false, 0.01f, "float");	cmd.add(paramArgAWGN);
      ValueArg<float> paramArgCx			("","cx",		"(Optional Advanced) Seed circle/sphere x position",					false, -1.0f, 	"float");	cmd.add(paramArgCx);
      ValueArg<float> paramArgCy			("","cy",		"(Optional Advanced) Seed circle/sphere y position",					false, -1.0f, 	"float");	cmd.add(paramArgCy);
      ValueArg<float> paramArgCz			("","cz",		"(Optional Advanced) Seed sphere z position",							false, -1.0f, 	"float");	cmd.add(paramArgCz);
      ValueArg<float> paramArgCr			("","cr",		"(Optional Advanced) Seed circle/sphere radius",						false, -1.0f, 	"float");	cmd.add(paramArgCr);
      ValueArg<float> paramArgIter		("","maxiter", "(Optional Advanced) Maximum iterations",									false, -1.0f, 	"float");	cmd.add(paramArgIter);

      // Basic parameters
      ValueArg<float> paramArgGrow		("g","grow",		"Prefer to shrink or grow",	false,	0.05f, 	"float");	cmd.add(paramArgGrow);
      ValueArg<float> paramArgSmooth	("s","smooth",		"Smoothing weight",				false, 	50.0f, 	"float");	cmd.add(paramArgSmooth);
      ValueArg<float> paramArgRange		("r","range",		"Segmentation range",			false,	3.0f, 	"float");	cmd.add(paramArgRange);

      ValueArg<int> 	  specDevArg		("","device",		"Choose specific device id rather than max flops [0, 1, ...]", 	false, 	-1, 			"int"); 		cmd.add(specDevArg);
      ValueArg<string> deviceArg			("d","type",		"Device type [CPU, GPU, ALL, ACCELERATOR, DEFAULT]", 					false,	"DEFAULT", 	"string"); 	cmd.add(deviceArg);
      ValueArg<string> phiArg				("","phi",			"Optional Tiff image for initial phi seed region [path/phi.tif]",	false,	"", 			"string");	cmd.add(phiArg);
      ValueArg<string> outputArg  		("o","output",		"Optional Save Tiff [path/filename.tif]",									false,	"out.tif",  "string");	cmd.add(outputArg);
      ValueArg<string> platformArg		("p","platform",	"Platform ID string [intel, nvidia, default, ...]", 					true,		"default", 	"string");  cmd.add(platformArg);
      ValueArg<string> imageArg			("i","image",		"Tiff image [path/image.tif]",												true,		"", 			"string");	cmd.add(imageArg);

      // Parse the args.
      cmd.parse(argc, argv);

      if (paramArgRange.isSet())
         acParams.push_back(paramArgRange.getValue());
      else
         acParams.push_back(paramArgSigma.getValue());

      acParams.push_back(paramArgTimestep.getValue());
      acParams.push_back(paramArgMu.getValue());

      if (paramArgSmooth.isSet())
         acParams.push_back(paramArgSmooth.getValue());
      else
         acParams.push_back(paramArgNu.getValue());

      acParams.push_back(paramArgAlf.getValue());

      if (paramArgGrow.isSet())
      {
         acParams.push_back(1.0f+max(0.0f, -paramArgGrow.getValue())); // paramArgLambda1.getValue());
         acParams.push_back(1.0f+max(0.0f,  paramArgGrow.getValue()));
      }
      else
      {
         acParams.push_back(paramArgLambda1.getValue());
         acParams.push_back(paramArgLambda2.getValue());
      }
      acParams.push_back(paramArgCx.getValue());
      acParams.push_back(paramArgCy.getValue());
      acParams.push_back(paramArgCz.getValue());
      acParams.push_back(paramArgCr.getValue());
      acParams.push_back(paramArgIter.getValue());
      acParams.push_back(paramArgAWGN.getValue());

      image.filename  = imageArg.getValue();
      outputFilename = outputArg.getValue();
      phi.filename = phiArg.getValue();
      string typeStr = deviceArg.getValue(); 		transform(typeStr.begin(), typeStr.end(), typeStr.begin(), ::toupper);
      platformVendor = platformArg.getValue();	transform(platformVendor.begin(), platformVendor.end(), platformVendor.begin(), ::tolower);
      specificDevice = specDevArg.getValue();

      // Select device type
      if (typeStr == "CPU") { cout << "Using CL_DEVICE_TYPE_CPU" << endl; type = CL_DEVICE_TYPE_CPU; }
      else if (typeStr == "GPU") { cout << "Using CL_DEVICE_TYPE_GPU" << endl; type = CL_DEVICE_TYPE_GPU; }
      else if (typeStr == "ALL") { cout << "Using CL_DEVICE_TYPE_ALL" << endl; type = CL_DEVICE_TYPE_ALL; }
      else if (typeStr == "ACCELERATOR") { cout << "Using CL_DEVICE_TYPE_ACCELERATOR" << endl; type = CL_DEVICE_TYPE_ACCELERATOR; }
      else
      { cout << "Using CL_DEVICE_TYPE_DEFAULT" << endl; }

      if (phi.filename != "")
      {
         cout << "Loading phi... ";
         phi.loadTif();
         cout << " done!" << endl;
      }

      image.loadTif();
      image.filename = outputFilename;

   }
   catch (ArgException &e) {}

   // Initialise the library
   if (!glfwInit())
      return -1;

   // Create a windowed mode window and its OpenGL context
   int w = 800, h = 600;
   window = glfwCreateWindow(w, h, "Interactive GPU Active Contours for Segmenting Inhomogeneous Objects.", NULL, NULL);
   if (!window)
   {
      glfwTerminate();
      return -1;
   }
   glfwSetWindowSizeCallback(window, windowSizeCallback);
   glfwSetScrollCallback(window, scrollCallback);
   glfwMakeContextCurrent(window);
   windowSizeCallback(window, w, h);
   glFinish();

   glewExperimental=GL_TRUE;
   GLenum err = glewInit();

   if (GLEW_OK != err)
   {
      fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      exit(1);
   }

   // Setup the shared opengl context for this window
   compute.setup(type, platformVendor, specificDevice);

   // Build the program
   renderer = unique_ptr<ActiveContour>(new ActiveContour);
   ActiveContour *ac = dynamic_cast<ActiveContour*>(renderer.get());
   ac->image = &image;
   ac->phi = &phi;
   renderer->program.build(compute, "kernels/kernel3D.cl");

   // Setup the program
   renderer->initialize(compute, w, h);
   renderer->setup(window, compute, acParams);

   // Loop until the user closes the window
   while (!glfwWindowShouldClose(window))
   {
      renderer->update(window, compute, image.width, image.height);
      glfwSwapBuffers(window);
      glfwPollEvents();
   }

   renderer->shutdown(compute);

   glfwTerminate();
   return 0;
}
