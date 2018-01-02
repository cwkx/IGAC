#ifndef headers_hpp_
#define headers_hpp_

// Dependencies
#include "cg_defines.hpp"
#include <GL/glew.h>
#ifdef __linux
#define GLFW_EXPOSE_NATIVE_GLX
#include <GL/glx.h>
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <GL/gl.h>
#include <CL/cl.hpp>
#include "tiffio.h"
#include "tclap/CmdLine.h"

// STL
#include <vector>
#include <string>
#include <iostream>
#include <limits>
#include <fstream>
#include <streambuf>
#include <memory>
#include <math.h>
#include <random>

#undef min
#undef max

// Namespaces
using namespace std;
using namespace TCLAP;

// User
#include "cg_image.hpp"
#include "cg_compute.hpp"
#include "cg_renderer.hpp"
#include "cg_active_contour.hpp"

#endif
