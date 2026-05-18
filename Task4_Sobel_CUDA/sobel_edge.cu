#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "lodepng.h"

#define MAX_IMAGES 100
#define MAX_PATH 256

__global__ void sobelKernel(unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int gxKernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int gyKernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    float gx = 0.0f;
    float gy = 0.0f;

    for (int ky = -1; ky <= 1; ky++)
    {
        for (int kx = -1; kx <= 1; kx++)
        {
            int px = x + kx;
            int py = y + ky;

            float gray = 0.0f;

            if (px >= 0 && px < width && py >= 0 && py < height)
            {
                int pixelIndex = (py * width + px) * 4;

                unsigned char r = input[pixelIndex];
                unsigned char g = input[pixelIndex + 1];
                unsigned char b = input[pixelIndex + 2];

                gray = 0.299f * r + 0.587f * g + 0.114f * b;
            }

            gx += gray * gxKernel[ky + 1][kx + 1];
            gy += gray * gyKernel[ky + 1][kx + 1];
        }
    }

    float magnitude = sqrtf((gx * gx) + (gy * gy));

    if (magnitude > 255.0f)
        magnitude = 255.0f;

    int outputIndex = (y * width + x) * 4;

    output[outputIndex] = (unsigned char)magnitude;
    output[outputIndex + 1] = (unsigned char)magnitude;
    output[outputIndex + 2] = (unsigned char)magnitude;
    output[outputIndex + 3] = 255;
}

void processImage(const char *inputFile, const char *outputFile)
{
    unsigned char *hostInputImage = NULL;
    unsigned char *hostOutputImage = NULL;
    unsigned width, height;

    unsigned error = lodepng_decode32_file(&hostInputImage, &width, &height, inputFile);

    if (error)
    {
        printf("Error loading %s: %s\n", inputFile, lodepng_error_text(error));
        return;
    }

    size_t imageSize = width * height * 4 * sizeof(unsigned char);

    hostOutputImage = (unsigned char *)malloc(imageSize);

    if (hostOutputImage == NULL)
    {
        printf("CPU memory allocation failed for output image.\n");
        free(hostInputImage);
        return;
    }

    unsigned char *deviceInputImage = NULL;
    unsigned char *deviceOutputImage = NULL;

    cudaMalloc((void **)&deviceInputImage, imageSize);
    cudaMalloc((void **)&deviceOutputImage, imageSize);

    cudaMemcpy(deviceInputImage, hostInputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numberOfBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    sobelKernel<<<numberOfBlocks, threadsPerBlock>>>(deviceInputImage, deviceOutputImage, width, height);

    cudaDeviceSynchronize();

    cudaMemcpy(hostOutputImage, deviceOutputImage, imageSize, cudaMemcpyDeviceToHost);

    error = lodepng_encode32_file(outputFile, hostOutputImage, width, height);

    if (error)
    {
        printf("Error saving %s: %s\n", outputFile, lodepng_error_text(error));
    }
    else
    {
        printf("Processed: %s -> %s\n", inputFile, outputFile);
    }

    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    free(hostInputImage);
    free(hostOutputImage);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s input1.png input2.png ... output_folder\n", argv[0]);
        return 1;
    }

    const char *outputFolder = argv[argc - 1];

    for (int i = 1; i < argc - 1; i++)
    {
        char outputFile[MAX_PATH];

        sprintf(outputFile, "%s/edge_%d.png", outputFolder, i);

        processImage(argv[i], outputFile);
    }

    printf("Sobel Edge Detection completed.\n");

    return 0;
}
