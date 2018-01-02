#include "cg_headers.hpp"

void Image::loadTif() {
	TIFF *tif = TIFFOpen(filename.c_str(), "r");
	depth = 0;
	rgba.clear();

    if (tif)
    {
    	uint32 bps;
    	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    	if (bps > 8)
		{
			cout << "Please ensure input TIFF is 8-bit! " << endl;
			// std::exit(-1);
		}

		do
		{
			depth++;

			size_t npixels;
			uint32* raster;
			vector<float> tmp;

			TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
			TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

			npixels = width * height;
			tmp.resize(width*height);

			raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
			if (raster != NULL) {
				if (TIFFReadRGBAImage(tif, width, height, raster, 0)) {
					for (int i=0; i<width*height; ++i)
					{
						tmp[i] = raster[i];
					}
				}
				_TIFFfree(raster);
			}

			rgba.insert(rgba.end(), tmp.begin(), tmp.end());

		} while (TIFFReadDirectory(tif));

        TIFFClose(tif);
    }

    cout << "Read image successfully with: " << width << ", " << height << ", " << depth << endl;
}

void Image::saveTif() const
{
	TIFF *out = TIFFOpen(filename.c_str(), "w");
	if (!out)
	{
		cout << "Could not open " << filename << " for writing" << endl;
		return;
	}
	unsigned short spp = 1; /* Samples per pixel */
	unsigned short bpp = 32; /* Bits per sample */
	unsigned short photo = PHOTOMETRIC_MINISBLACK;

	for (int page = 0; page < depth; page++)
	{
		TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width / spp);
		TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
		TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bpp);
		TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, spp);
		TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
		TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(out, TIFFTAG_PHOTOMETRIC, photo);
		TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT);
		TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
		TIFFSetField(out, TIFFTAG_PAGENUMBER, page, depth);

		for (int j = 0; j<height; ++j)
		{
			const int flip = (height-j)-1;
			TIFFWriteScanline(out, (void*)&rgba[page * height * width + flip * width], j, 0);
		}

		TIFFWriteDirectory(out);
	}

	TIFFClose(out);

	cout << "Saved " << filename << " successfully!" << endl;

	return;
}

void Image::normalize() {
	float max = numeric_limits<float>::min();
	float min = numeric_limits<float>::max();

	for (unsigned int i = 0; i < rgba.size(); ++i) {
		min = std::min(rgba[i], min);
		max = std::max(rgba[i], max);
	}

	for (unsigned int i = 0; i < rgba.size(); ++i)
		rgba[i] = (rgba[i] - min) / (max - min);
}

void Image::spread(int amount) {
	vector<float> spread;

	for (unsigned int i = 0; i < rgba.size(); ++i)
		for (int a = 0; a < amount; ++a)
			spread.push_back(rgba[i]);

	rgba = spread;
}

void Image::awgn(float amount)
{
	const double mean = 0.0;
	const double stddev = amount;
	default_random_engine generator;
	normal_distribution<double> dist(mean, stddev);

	for (unsigned int i = 0; i < rgba.size(); ++i)
		rgba[i] = rgba[i] + dist(generator);
}
