#include "cg_headers.hpp"

void Renderer::initialize(const Compute &compute, int width, int height)
{
   glfwSwapInterval(0);
   glDisable(GL_DEPTH_TEST);
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   unsigned int size = 4096*4096*4; // BAD!

   glGenBuffers(1, &pboID);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
   glBufferData(GL_PIXEL_UNPACK_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

   pbo = cl::BufferGL(compute.context, CL_MEM_WRITE_ONLY, pboID);
   mem.push_back(pbo);
   glFinish();
}

void Renderer::resize(const Compute &compute, int width, int height)
{
   // cout << "Resize!" << endl;
}
