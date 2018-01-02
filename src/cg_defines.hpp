#if defined(__APPLE__) || defined(__MACOSX)
	#define GLFW_EXPOSE_NATIVE_COCOA
	#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
	#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
  #ifdef _WIN32
	#define GLFW_EXPOSE_NATIVE_WGL
	#define GLFW_EXPOSE_NATIVE_WIN32
  #else
	#define GLFW_EXPOSE_NATIVE_X11
  #endif
#endif
#define __CL_ENABLE_EXCEPTIONS
#define GLEW_STATIC
