struct Renderer
{
   Program program;
   GLuint pboID;
   cl::BufferGL pbo;
   vector<cl::Memory> mem;
   vector<cl::Kernel> kernels;
   int arg=0;

   // Push arguments to last kernel and reset arg (in the variadic case)
   template<typename T>
   void push(const T& t) { kernels.back().setArg(arg++,t); }
   template<typename U, typename... T>
   void push(const U& head, const T&... tail) { push(head); push(tail...); arg=0; }

   // Prepare a kernel and map its memory
   template<typename... T>
   void map(const string& name, const T&... tail)
   {
      arg=0;
      kernels.push_back(cl::Kernel(program.data,name.c_str()));
      push(tail...);
   }

   void 			 initialize	(const Compute &compute, int width, int height);
   void 			 resize		(const Compute &compute, int width, int height);
   virtual void setup		(GLFWwindow* window, const Compute &compute, const vector<float> &params) = 0;
   virtual void update		(GLFWwindow* window, const Compute &compute, int width, int height) = 0;
   virtual void scroll		(double xOffset, double yOffset) = 0;
   virtual void shutdown	(const Compute &compute) = 0;
   virtual 		~Renderer	() {}
};
