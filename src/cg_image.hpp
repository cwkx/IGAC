struct Image
{
   int width;
   int height;
   int depth;
   vector<float> rgba;	// floating point (4-channel)
   string filename; 		// set before calling load and save tif

   void loadTif();
   void saveTif() const;
   void normalize();
   void spread(int amount=4);
   void awgn(float amount);
};
