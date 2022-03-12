/*
	Parallel Programming
	Exercise 2
	Mandelbrot
	Thomas Bründl
	Thomas Stummer
*/

#include <omp.h>
#include <stdio.h>
#include <ostream>
#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <thread>
#include "../PPR_Mandelbrot/tga.h"

using namespace std::chrono;
using namespace std;
using namespace tga;

const int BYTES_PER_PIXEL = 3;
const int GL_RGB = 0;
const int BITS_PER_PIXEL = 24;
const unsigned int MAX_COLOR_CODE = 255 * 3;

tuple<float, float> normalizeToViewRectangle(
	int pX, int pY, float minX, float minY, float maxX, float maxY, 
	unsigned int w, unsigned int h)
{
	float cX = ((maxX - minX) * ((float)pX / (float)w)) + minX;
	float cY = ((maxY - minY) * ((float)pY / (float)h)) + minY;
	return { cX, cY };
}

unsigned int calcPix(
	int pX, int pY, float minX, float minY, float maxX, float maxY,
	unsigned int maxIterations, unsigned int w, unsigned int h)
{
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
			return n * MAX_COLOR_CODE / maxIterations;
		}
		zX = x;
		zY = y;
	}
	return 0;
}

void parallelSolution(
	float minX, float minY, float maxX, float maxY,
	unsigned int maxIterations, unsigned int w, unsigned int h,
	int num_threads)
{
	vector<unsigned char> v (w * h * BYTES_PER_PIXEL);

	omp_set_num_threads(num_threads);
	auto startTimeStamp = high_resolution_clock::now();

#pragma omp parallel
	{
		for (int y = omp_get_thread_num(); y < h; y += omp_get_num_threads())
		{
			for (int x = 0; x < w; x++)
			{
				unsigned int color = calcPix(x, y, minX, minY, maxX, maxY, maxIterations, w, h);
				unsigned int p = (y * w + x) * BYTES_PER_PIXEL;
				v[p] = (unsigned char) color % 255;
				v[p + 1] = (unsigned char) color % (255 * 255);
				v[p + 2] = (unsigned char) color % (255 * 255 * 255);
			}
		}
	}

	auto stopTimeStamp = high_resolution_clock::now();
	auto durationMS = duration_cast<microseconds>(stopTimeStamp - startTimeStamp).count() / 1000;
	cout << "Parallel mandelbrot using " << num_threads << " threads took: " << durationMS << " ms." << endl;

	auto fileName = "mandelbrot_parallel_" + to_string(num_threads) + ".tga";
	tga::TGAImage image = { v, BITS_PER_PIXEL, w, h, GL_RGB };
	tga::saveTGA(image, &fileName[0]);
}

void serialSolution(
	float minX, float minY, float maxX, float maxY,
	unsigned int maxIterations, unsigned int w, unsigned int h)
{
	vector<unsigned char> v (w * h * BYTES_PER_PIXEL);
	auto startTimeStamp = high_resolution_clock::now();

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			unsigned int color = calcPix(x, y, minX, minY, maxX, maxY, maxIterations, w, h);
			unsigned int p = (y * w + x) * BYTES_PER_PIXEL;
			v[p] = (unsigned char) color % 255;
			v[p + 1] = (unsigned char) color % (255 * 255);
			v[p + 2] = (unsigned char) color % (255 * 255 * 255);
		}
	}

	auto stopTimeStamp = high_resolution_clock::now();
	auto durationMS = duration_cast<microseconds>(stopTimeStamp - startTimeStamp).count() / 1000;
	cout << "Serial mandelbrot took: " << durationMS << " ms." << endl;

	TGAImage image = { v, BITS_PER_PIXEL, w, h, GL_RGB };
	saveTGA(image, "mandelbrot_serial.tga");
}

int main(int argc, char* argv[])
{
	unsigned int maxIterations = 1000;
	unsigned int w = 1024;
	unsigned int h = 1024;
	float minX = -1.5;
	float minY = -1;
	float maxX = .5;
	float maxY = 1;

	serialSolution(minX, minY, maxX, maxY, maxIterations, w, h);

	int coresOnMachine = thread::hardware_concurrency();

	if (coresOnMachine > 1)
	{
		for (int cores = 2; cores <= coresOnMachine; cores++)
		{
			parallelSolution(minX, minY, maxX, maxY, maxIterations, w, h, cores);
		}
	}
	else
	{
		cout << "WARNING: This machine has only one core. No parallel calculation is performed!" << endl;
	}

	return 0;
}
