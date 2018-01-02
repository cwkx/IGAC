#include "cg_headers.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void ActiveContour::mapParam(int i, float &x, string &name, bool set)
{
   // Sigma case is a bit special as we don't store sigma anywhere, just the coefficients vector
   if (i == 0)
   {
      if (set)
      {
         sigma=x;
         setGauss();
      }
      x = sigma;
      name="Sigma";
   }

   // Get/Set remaining parameters
   if (i == 1) { if (set) params.s[2] = x; x = params.s[2]; name="nu"; }
   if (i == 2)
   {
      if (set)
      {
         slambda = x;
         lambda.s[0] = 1.0f+max(0.0f, -x);
         lambda.s[1] = 1.0f+max(0.0f,  x);
      }
      x = slambda;
      name = "Lambda";
   }

   if (i == 3) { if (set) lambda.s[0] = x; x = lambda.s[0]; name="Lambda1"; }
   if (i == 4) { if (set) lambda.s[1] = x; x = lambda.s[1]; name="Lambda2"; }
   if (i == 5) { if (set) params.s[0] = x; x = params.s[0]; name="Timestep"; }
   if (i == 6) { if (set) params.s[1] = x; x = params.s[1]; name="mu"; }
   if (i == 7) { if (set) params.s[3] = x; x = params.s[3]; name="alf"; }
}

void ActiveContour::setParams(GLFWwindow* window, const Compute &compute, double ypos, int windowHeight, bool &changed, bool &reinitialize)
{
   static bool   key[8];		// keyboard keys 1-8
   static double mouseY[8]; 	// previous mouse y-positions
   static float  prev[8];  	// previous values

   changed = false;
   reinitialize = false;
   int keyRange = 3+5*glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);

   for (int i=0; i<keyRange; ++i)
   {
      // key pressed
      if (key[i] == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_1+i) == GLFW_PRESS)
      {
         mouseY[i] = ypos;

         string name;
         float x;
         mapParam(i, x, name,  false);
         prev[i] = x;
      }
   }

   for (int i=0; i<keyRange; ++i)
   {
      if (key[i] == GLFW_PRESS)
      {
         changed = true;
         if (i==0)
            reinitialize = true;

         float speed = 1.0f;

         if (i==0) speed = 5.0f;  // sigma
         if (i==1) speed = 10.0f; // nu
         if (i==2) speed = 1.0f;  // lambda
         if (i==3) speed = 1.0f;  // lambda1
         if (i==4) speed = 1.0f;  // lambda2
         if (i==5) speed = 0.5f;  // timestep
         if (i==6) speed = 0.1;   // mu
         if (i==7) speed = 10.0f; // alf

         // Mapping
         string name;
         float x = prev[i] - speed*(ypos-mouseY[i])/windowHeight;
         mapParam(i, x, name, true);

         // cout << name << ": " << i+1 << " adjusted: " << x << endl;
      }

      key[i] = glfwGetKey(window, GLFW_KEY_1+i);
   }

   // todo: "changed" check as this is a bit slow...
}

void ActiveContour::save(const Compute &compute, const cl::Image &im, const int channel)
{
   cl::size_t<3> origin; origin[0] = 0; origin[1] = 0, origin[2] = 0;
   cl::size_t<3> region; region[0] = image->width; region[1] = image->height; region[2] = image->depth;

   compute.queue.enqueueReadImage(im, true, origin, region, 0, 0, &image->rgba[0]);

   for (unsigned int i=0; i<image->rgba.size()/4; ++i)
   {
      image->rgba[i] = image->rgba[i*4+channel];
   }

   image->rgba.resize(image->width*image->height*image->depth);
   image->saveTif();
}

void ActiveContour::setup(GLFWwindow* window, const Compute &compute, const vector<float> &acParams)
{
   sigma = acParams[0];
   slambda = 0.05;
   setGauss();
   image->normalize();
   image->awgn(acParams[12]);
   image->spread(4);
   maxIterations = acParams[11];

   try
   {
      K = cl::Buffer(compute.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*gauss.size(), &gauss[0]);
      A = cl::Image3D(compute.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_FLOAT), image->width, image->height, image->depth, 0, 0, &image->rgba[0]);
      C = cl::Image3D(compute.context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), image->width, image->height, image->depth, 0, 0, NULL);

      // Prepare phi from previous iteration
      if (phi->filename != "")
      {
         phi->normalize();
         phi->spread(4);

     B = cl::Image3D(compute.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_FLOAT), phi->width, phi->height, phi->depth, 0, 0, &phi->rgba[0]);
      }
      else
         B = cl::Image3D(compute.context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), image->width, image->height, image->depth, 0, 0, NULL);


      cl::size_t<3> origin; origin[0] = 0; origin[1] = 0, origin[2] = 0;
      cl::size_t<3> region; region[0] = image->width; region[1] = image->height; region[2] = image->depth;

      params = {acParams[1], acParams[2], acParams[3], acParams[4]};
      lambda = {acParams[5], acParams[6]};

      compute.queue.enqueueWriteBuffer(K, false, 0, gauss.size(), &gauss[0], NULL);
      compute.queue.enqueueWriteImage	(A, false, origin, region, 0, 0, &image->rgba[0]);

      int windowWidth, windowHeight;
      glfwGetWindowSize(window, &windowWidth, &windowHeight);

      int w = image->width;
      int h = image->height;
      int d = image->depth;
      kernels.clear();
      timings.resize(11);
      cl_float16 mat;
      activeSlice = d/2;

      map("prepare", 		A,C);
      map("horzGausA", 	C,B,K,(int)gauss.size());
      map("vertGausA", 	B,C,K,(int)gauss.size());
      map("depthGausA", 	C,B,K,(int)gauss.size());
      map("compose", 		A,B,C, acParams[7] >= 0.0f ? acParams[7] : (float)image->width/2,
          acParams[8] >= 0.0f ? image->height-acParams[8] : (float)image->height/2,
          acParams[9] >= 0.0f ? acParams[9] : (float)image->depth/2,
          acParams[10], 0); // set phi here
      map("neumannCopyA", C,B,A,0,w,h,d);

      // Launch first 5 prepapre kernels
      for (int i=0; i<5; ++i)
         compute.queue.enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(image->width, image->height, image->depth), cl::NullRange);

      if (phi->filename != "")
      {
         kernels[5].setArg(3, 1);
         compute.queue.enqueueWriteImage	(B, true, origin, region, 0, 0, &phi->rgba[0]);
         compute.queue.enqueueNDRangeKernel(kernels[5], cl::NullRange, cl::NDRange(image->width, image->height, image->depth), cl::NullRange);
         kernels[5].setArg(3, 0);
      }
      else
         compute.queue.enqueueNDRangeKernel(kernels[5], cl::NullRange, cl::NDRange(image->width, image->height, image->depth), cl::NullRange);

      map("normalisedGradPhi",	A,C,w,h,d);
      map("divPrepFirstFilter", 	A,C,B);
      map("horzGausB", 			B,C,K,(int)gauss.size());
      map("vertGausB", 			C,B,K,(int)gauss.size());
      map("depthGausB", 			B,C,K,(int)gauss.size());
      map("prepSecondFilter", 	A,C,B, lambda);
      map("horzGausC", 			B,C,K,(int)gauss.size());
      map("vertGausC", 			C,B,K,(int)gauss.size());
      map("depthGausC", 			B,C,K,(int)gauss.size());
      map("updatePhi", 			A,C,B,w,h,d,lambda, params);
      map("neumannCopyB", 		B,A,w,h,d);
      map("render", 				A,pbo,windowWidth, windowHeight, w,h,d, d/2, mat, 0.0f, 0, 0);

   }
   catch (cl::Error &err)
   {
      cout << "Could not setup buffers: " << err.what() << " (" << err.err() << ")" << endl;
   }
}

void middleDragging(GLFWwindow* window, int &activeSlice, int maxDepth, float mouseNormX, float mouseNormY)
{
   static int middleState = 0;
   int lastMiddle = middleState;
   middleState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);
   static float seedMiddleY = 0;
   static int   seedActiveState = 0;
   if (lastMiddle == GLFW_RELEASE && middleState == GLFW_PRESS)
   {
      seedMiddleY = mouseNormY;
      seedActiveState = activeSlice;
   }
   if (middleState == GLFW_PRESS)
   {
      activeSlice = seedActiveState + (mouseNormY-seedMiddleY)*maxDepth*3;
      activeSlice = activeSlice > maxDepth ? maxDepth : activeSlice;
      activeSlice = activeSlice < 0 ? 0 : activeSlice;
   }
}

cl_float3 cross(const cl_float3 x, const cl_float3 z)
{
   return {z.s[1]*x.s[2]-z.s[2]*x.s[1], z.s[2]*x.s[0]-z.s[0]*x.s[2], z.s[0]*x.s[1]-z.s[1]*x.s[0]};
}

inline cl_float squaredLength(const cl_float3 v)
{
   return v.s[0]*v.s[0] + v.s[1]*v.s[1] + v.s[2]*v.s[2];
}

inline cl_float length(const cl_float3 v)
{
   return sqrt(squaredLength(v));
}

inline cl_float3 normalise(const cl_float3 v)
{
   const float n = length(v);
   return { v.s[0]/n, v.s[1]/n, v.s[2]/n };
}

inline cl_float3 neg(const cl_float3 v)
{
   return {-v.s[0], -v.s[1], -v.s[2]};
}

inline cl_float3 div(const cl_float3 v, const float w)
{
   return {v.s[0]/w, v.s[1]/w, v.s[2]/w};
}

void ActiveContour::update(GLFWwindow* window, const Compute &compute, int width, int height)
{
   // Input states
   static int rightState = 0;
   int lastState = rightState;
   rightState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
   int leftState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
   double xpos, ypos;
   int windowWidth, windowHeight;
   glfwGetCursorPos(window, &xpos, &ypos);
   glfwGetWindowSize(window, &windowWidth, &windowHeight);
   float mouseNormX = (float)xpos/windowWidth;
   float mouseNormY = (float)ypos/windowHeight;
   static float seedNormX = 0.0f;
   static float seedNormY = 0.0f;
   int modellingState = 1;
   if (glfwGetKey(window, GLFW_KEY_SPACE)) 		modellingState = 0;
   if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) 	modellingState = 2;
   if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL))	modellingState = 3;

   // 3d rotation and control states
   static bool lastAlt = false;
   static bool lastO = false;
   static bool toggleO = false;
   static bool lastI = false;
   static bool toggleI = false;
   static float tx = 0.125f; // to make it start at a nice place
   static float ty = 0.304086723984696; // (1/2 phi - acos(2/sqrt(6))/pi) to make it start at a nice place
   static float pressSeedX = 0.0f;
   static float pressSeedY = 0.0f;
   static float alpha = 0.0f;
   static float adelta = 0.0f;
   static float aortho = 0.0f;
   static float astretch = 0.0f;
   static float timeLast = glfwGetTime();

   // alpha blend state
   float timeDelta = 0.01;//glfwGetTime()-timeLast; // CHANGE FOR RECORDING!!
   timeLast = glfwGetTime();
   if (glfwGetKey(window, GLFW_KEY_LEFT_ALT))
      alpha += 5*timeDelta;
   else
      alpha -= 5*timeDelta;

   if (glfwGetKey(window, GLFW_KEY_D))
      adelta += 5*timeDelta;
   else
      adelta -= 5*timeDelta;

   if (!lastO && glfwGetKey(window, GLFW_KEY_O)) toggleO = !toggleO;
   if (!lastI && glfwGetKey(window, GLFW_KEY_I)) toggleI = !toggleI;

   lastO = glfwGetKey(window, GLFW_KEY_O);
   lastI = glfwGetKey(window, GLFW_KEY_I);

   if (toggleO) aortho   += 5*timeDelta;
   else		 aortho   -= 5*timeDelta;
   if (toggleI) astretch += 5*timeDelta;
   else		 astretch -= 5*timeDelta;

   alpha  = max(min(alpha,0.99f), -0.001f);
   adelta = max(min(adelta,0.99f), -0.001f);
   aortho = max(min(aortho,0.99f), 0.0f);
   astretch = max(min(astretch,0.99f), 0.0f);

   if (!lastAlt && glfwGetKey(window, GLFW_KEY_LEFT_ALT))
   {
      // press ALT
      pressSeedX = mouseNormX;
      pressSeedY = mouseNormY;

      glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

   }
   else if (lastAlt && !glfwGetKey(window, GLFW_KEY_LEFT_ALT))
   {
      // release ALT
      tx -=  (mouseNormX-pressSeedX);
      ty -=  (mouseNormY-pressSeedY);

      glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
   }
   else if (glfwGetKey(window, GLFW_KEY_LEFT_ALT))
   {
      // hold ALT
      const float dx =  tx-(mouseNormX-pressSeedX);
      const float dy =  ty-(mouseNormY-pressSeedY);

      const float pi = 3.14159265359;
      const float phi = (dx-0.5f) * 2 * pi;
      float theta = dy * pi;

      theta = max(min(theta, 0.9f*pi),0.001f);
      float viewDist = 1; // sin(glfwGetTime());
      cl_float3 camPos = {viewDist*sin(theta)*sin(phi),
                          viewDist*cos(theta),
                          viewDist*sin(theta)*cos(phi)
                         };	// x, y, z

      cl_float3 Z = div(neg(camPos), length(camPos));
      cl_float3 X = normalise(cross({ 0, 1, 0 },Z));
      cl_float3 Y = normalise(cross(Z,X));

      cl_float16 mat  = {X.s[0], X.s[1], X.s[2], 0,
                         Y.s[0], Y.s[1], Y.s[2], 0,
                         Z.s[0], Z.s[1], Z.s[2], 0,

           camPos.s[0], camPos.s[1], camPos.s[2], aortho
                        };

      kernels.back().setArg(8, mat);
   }
   kernels.back().setArg(9, alpha);
   kernels.back().setArg(10, adelta);
   kernels.back().setArg(11, astretch);
   lastAlt = glfwGetKey(window, GLFW_KEY_LEFT_ALT);

   // set keyboard params 1-9
   bool changed, reinitialize;
   setParams(window, compute, ypos, windowHeight, changed, reinitialize);

   // pause functionality
   static bool lastZ = false;
   static bool lastT = true;
   if (glfwGetKey(window, GLFW_KEY_Z) != lastZ)
   {
      lastZ = glfwGetKey(window, GLFW_KEY_Z);
      if (lastZ)
      {
         lastT = !lastT;
         params.s[0] = lastT ? 0.1f : 0.0f;
         changed = true;
      }
   }

   if (reinitialize)
   {
      K = cl::Buffer(compute.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*gauss.size(), &gauss[0]);
      compute.queue.enqueueWriteBuffer(K, false, 0, gauss.size(), &gauss[0], NULL);

      kernels[1] .setArg(2, K); kernels[1] .setArg(3, (int)gauss.size());
      kernels[2] .setArg(2, K); kernels[2] .setArg(3, (int)gauss.size());
      kernels[3] .setArg(2, K); kernels[3] .setArg(3, (int)gauss.size());

      // Launch first 6 prepapre kernels
      for (int i=0; i<6; ++i)
         compute.queue.enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(image->width, image->height, image->depth), cl::NullRange);

      kernels[8] .setArg(2, K); kernels[8] .setArg(3, (int)gauss.size());
      kernels[9] .setArg(2, K); kernels[9] .setArg(3, (int)gauss.size());
      kernels[10].setArg(2, K); kernels[10].setArg(3, (int)gauss.size());
      kernels[12].setArg(2, K); kernels[12].setArg(3, (int)gauss.size());
      kernels[13].setArg(2, K); kernels[13].setArg(3, (int)gauss.size());
      kernels[14].setArg(2, K); kernels[14].setArg(3, (int)gauss.size());
   }
   if (changed)
   {
      printParams(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT));
      kernels[11].setArg(3, lambda); kernels[15].setArg(6, lambda);
      kernels[15].setArg(7, params);
   }

   middleDragging(window, activeSlice, image->depth, mouseNormX, mouseNormY);

   kernels.back().setArg(7, activeSlice);
   kernels[4].setArg(5, (float)activeSlice);

   // mouse "params"
   if (lastState == GLFW_RELEASE && rightState == GLFW_PRESS)
   {
      seedNormX = mouseNormX;
      seedNormY = mouseNormY;
   }

   // Drag Phi
   if (leftState == GLFW_PRESS)
   {
      kernels[4].setArg(3, mouseNormX * width);
      kernels[4].setArg(4, height-(mouseNormY * height));

      kernels[4].setArg(7, modellingState);

      for (int i=4; i<6; ++i)
         compute.queue.enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(image->width, image->height, image->depth), cl::NullRange);
   }
   // Resize Phi
   if (rightState == GLFW_PRESS || modellingState == 0)
   {
      kernels[4].setArg(3, seedNormX * width);
      kernels[4].setArg(4, height-(seedNormY * height));
      kernels[4].setArg(6, sqrt((seedNormX-mouseNormX)*(seedNormX-mouseNormX)*width*width +
                                (seedNormY-mouseNormY)*(seedNormY-mouseNormY)*height*height)
                        *(modellingState!=0));
      kernels[4].setArg(7, modellingState);

      for (int i=4; i<6; ++i)
         compute.queue.enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(image->width, image->height, image->depth), cl::NullRange);
   }
   // Update Phi
   else
   {
      // Prepare events
      vector<cl::Event> events;
      events.resize(11);

      for (int i=6; i<17; ++i)
         compute.queue.enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(image->width, image->height, image->depth), cl::NullRange, NULL, &events[i-6]);

      cl::Event::waitForEvents(events);
      for (unsigned int i=0; i<events.size(); ++i)
      {
         cl_ulong startTimeNs = events[i].getProfilingInfo<CL_PROFILING_COMMAND_START>();
         cl_ulong endTimeNs   = events[i].getProfilingInfo<CL_PROFILING_COMMAND_END>();

         timings[i].push_back(0.000001f*(endTimeNs-startTimeNs));
         // cout << "kernel: " << 6+i << ", " << endl;
      }

      static int iter=0;
      if (maxIterations>0 && ++iter == maxIterations)
      {
         save(compute, A, 2);
         exit(EXIT_SUCCESS);
      }
   }

   if (glfwGetKey(window, GLFW_KEY_L))
      logTiming();
   if (glfwGetKey(window, GLFW_KEY_P))
      printParams(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT));
   if (glfwGetKey(window, GLFW_KEY_T))
   {
      cout << "Time: " << glfwGetTime() << endl;
   }

   // Save Phi
   if (glfwGetKey(window, GLFW_KEY_S) || glfwGetKey(window, GLFW_KEY_X))
   {
      save(compute, A, 2);
      if (glfwGetKey(window, GLFW_KEY_X))
         exit(EXIT_SUCCESS);
   }

   kernels.back().setArg(2, windowWidth);
   kernels.back().setArg(3, windowHeight);
   kernels.back().setArg(12, mouseNormX * windowWidth);
   kernels.back().setArg(13, windowHeight-(mouseNormY * windowHeight));

   // Render
   compute.queue.enqueueAcquireGLObjects(&mem, NULL, NULL);
   compute.queue.enqueueNDRangeKernel(kernels.back(), cl::NullRange, cl::NDRange(windowWidth, windowHeight), cl::NullRange, NULL, NULL);
   compute.queue.finish();
   compute.queue.enqueueReleaseGLObjects(&mem, NULL, NULL);
   compute.queue.finish();

   // Draw
   glClear(GL_COLOR_BUFFER_BIT);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
   glRasterPos2i(0, 0);
   glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, 0);

   // Record Renderer, pause functionality
   static bool lastR = false;
   static bool lastRT = false;
   if (glfwGetKey(window, GLFW_KEY_R) != lastR)
   {
      lastR = glfwGetKey(window, GLFW_KEY_R);
      if (lastR)
      {
         lastRT = !lastRT;
      }
   }
   if (lastRT)
   {
      static vector<unsigned char> rgba;
      if (rgba.size() < (unsigned int)(windowWidth * windowHeight * 4))
         rgba.resize(windowWidth * windowHeight * 4);

      glReadPixels(0,0, windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, &rgba[0]);

      for ( int i = 0; i < windowHeight/2; ++i)
      {
         int k = windowHeight - 1 - i;
         for (int j = 0; j<windowWidth*4; ++j)
         {
            unsigned char temp = rgba[i * windowWidth*4 + j];
            rgba[i * windowWidth*4 + j] = rgba[k * windowWidth*4 + j];
            rgba[k * windowWidth*4 + j] = temp;
         }
      }

      std::ostringstream oss;
      static int iter=0;
      oss << "movies/out" << iter++ << ".tga";

      stbi_write_tga(oss.str().c_str(),windowWidth,windowHeight, 4, &rgba[0]);

      cout << "Saved!" << endl;
   }

   // Unbind Buffer
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void ActiveContour::scroll(double x, double y)
{
   if (y < 0)
      activeSlice = min(activeSlice+1, image->depth);
   if (y > 0)
      activeSlice = max(activeSlice-1, 0);
}

void ActiveContour::setGauss()
{
   sigma = max(sigma, 0.00001f);
   int size = (4 * sigma + 1); // kernel dimension will be size x size
   if (size%2 == 0)
      size++;

   gauss = vector<float>(size);
   int centre = size/2; // index of centre pixel (deliberate rounding down)

   for (int i=0; i<size; ++i)
      gauss[i] = exp(-(i-centre)*(i-centre) / (2.0 * sigma * sigma));

   // normalise
   float sum = 0;
   for (int i = 0; i < size; ++i)
      sum += gauss[i];

   for (int i = 0; i < size; ++i)
      gauss[i] /= sum;

   // cout << "GAUSSIAN SIZE: " << gauss.size() << "\n";
}

void ActiveContour::logTiming()
{
   cout << "Logging timings... ";
   ofstream file ("timings.csv");
   if (file.is_open())
   {
      for (unsigned int i=0; i<timings.size(); ++i)
      {
         file << "kernel" << i << ",";

         auto first = true;
         for (float f : timings[i])
         {
            if (!first) { file << ","; }
            first = false;
            file << f;
         }

         file << "\n";
      }
   }
   file.close();
   cout << "Done!" << endl;
}

void ActiveContour::printParams(bool advanced)
{
   if (advanced)
   {
      cout << "| sigma: " 	  << left << setw(9) << setprecision(3) << sigma
           << "| nu: " 		  << left << setw(9) << setprecision(3) << params.s[2]
           << "| lambda: "	  << left << setw(9) << setprecision(3) << slambda
           << "| lambda1: "  << left << setw(9) << setprecision(3) << lambda.s[0]
           << "| lambda2: "  << left << setw(9) << setprecision(3) << lambda.s[1]
           << "| timestep: " << left << setw(9) << setprecision(3) << params.s[0]
           << "| mu: " 		  << left << setw(9) << setprecision(3) << params.s[1]
           << "| alf: " 	  << left << setw(9) << setprecision(3) << params.s[3]
           << "|" << endl;
   }
   else
   {
      cout << "| range: " 	<< left << setw(9) << setprecision(3) << sigma			<< ((sigma<1.0) ? "WARNING! Out of suggested range!" : "")
           << "| smooth: " << left << setw(9) << setprecision(3) << params.s[2]
           << "| grow: "	<< left << setw(9) << setprecision(3) << slambda 		<< ((slambda>0.1 || slambda<-0.1) ? "WARNING! Out of suggested range!" : "")
           << "|" << endl;
   }
}
