#include <omp.h>
#include <stdio.h>
#include <ostream>
#include <chrono>
#include <iostream>
#include <vector>
#include "../PPR_Mandelbrot/tga.h"
#include <tuple>

using namespace std::chrono;
using namespace std;

std::tuple<float, float> normalizeToViewRectangle(int pX, int pY, float minX, float minY, float maxX, float maxY, unsigned int w, unsigned int h)
{
	float cX = ((maxX - minX) * ((float)pX / (float)w)) + minX;
	float cY = ((maxY - minY) * ((float)pY / (float)h)) + minY;
	return { cX, cY };
}

unsigned char calcPix(int pX, int pY, float minX, float minY, float maxX, float maxY,
	int maxIterations, unsigned int w, unsigned int h)
{

	const unsigned char maxColorCode = 255;
	float cX;
	float cY;

	tie(cX, cY) = normalizeToViewRectangle(pX, pY, minX, minY, maxX, maxY, w, h);

	float zX = cX;
	float zY = cY;
	for (int n = 0; n < maxIterations; n++)
	{
		float x = (zX * zX - zY * zY) + cX;
		float y = (zY * zX + zX * zY) + cY;
		if ((x * x + y * y) > 4)
		{
			return (unsigned char)(n * maxColorCode / maxIterations);
		}
		zX = x;
		zY = y;
	}
	return 0;
}

void parallelSolution(float minX, float minY, float maxX, float maxY,
	int maxIterations, unsigned int w, unsigned int h)
{

	int bytesPerPixel = 3;
	int GL_RGB = 0;
	int Bpp = 24;

	std::vector<unsigned char> v(w * h * bytesPerPixel);

	omp_set_num_threads(12);
	auto startTimeStamp = high_resolution_clock::now();
#pragma omp parallel
	{
		for (int y = omp_get_thread_num(); y < h; y += omp_get_num_threads())
		{
			for (int x = 0; x < w; x++)
			{
				unsigned char color = calcPix(x, y, minX, minY, maxX, maxY, maxIterations, w, h);
				unsigned int p = (y * w + x) * bytesPerPixel;
				v[p] = color;
				v[p + 1] = color;
				v[p + 2] = color;
			}
		}
	}

	auto stopTimeStamp = high_resolution_clock::now();
	auto durationMS = duration_cast<microseconds>(stopTimeStamp - startTimeStamp).count() / 1000;
	std::cout << "Computation_Parallel: " << durationMS << " ms" << std::endl;

	tga::TGAImage image = { v, Bpp, w, h, GL_RGB };
	tga::saveTGA(image, "image_p.tga");
}

void serialSolution(float minX, float minY, float maxX, float maxY,
	int maxIterations, unsigned int w, unsigned int h)
{

	std::vector<unsigned char> v(w * h * 3);

	auto startTimeStamp = high_resolution_clock::now();

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			unsigned char color = calcPix(x, y, minX, minY, maxX, maxY, maxIterations, w, h);
			unsigned int p = (y * w + x) * 3;
			v[p] = color;
			v[p + 1] = color;
			v[p + 2] = color;
		}
	}

	auto stopTimeStamp = high_resolution_clock::now();
	auto durationMS = duration_cast<microseconds>(stopTimeStamp - startTimeStamp).count() / 1000;
	std::cout << "Computation_Serial: " << durationMS << " ms" << std::endl;

	tga::TGAImage image = { v, 24, w, h, 2 };
	tga::saveTGA(image, "image_s.tga");
}

int main(int argc, char* argv[])
{
	unsigned int w = 1024;
	unsigned int h = 1024;
	float minX = -1.5;
	float minY = -1;
	float maxX = .5;
	float maxY = 1;
	int maxIterations = 1000;

	parallelSolution(minX, minY, maxX, maxY, maxIterations, w, h);
	serialSolution(minX, minY, maxX, maxY, maxIterations, w, h);

	return 0;
}
