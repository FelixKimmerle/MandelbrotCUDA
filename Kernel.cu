#include "Kernel.h"
#include <thrust/complex.h>

#define BLOCK_SIZE 16

__device__ uchar4 convert_one_pixel_to_rgb(float h, float s, float v)
{
    float r, g, b;

    float f = h / 60.0f;
    float hi = floorf(f);
    f = f - hi;
    float p = v * (1 - s);
    float q = v * (1 - s * f);
    float t = v * (1 - s * (1 - f));

    if (hi == 0.0f || hi == 6.0f)
    {
        r = v;
        g = t;
        b = p;
    }
    else if (hi == 1.0f)
    {
        r = q;
        g = v;
        b = p;
    }
    else if (hi == 2.0f)
    {
        r = p;
        g = v;
        b = t;
    }
    else if (hi == 3.0f)
    {
        r = p;
        g = q;
        b = v;
    }
    else if (hi == 4.0f)
    {
        r = t;
        g = p;
        b = v;
    }
    else
    {
        r = v;
        g = p;
        b = q;
    }

    unsigned char red = 255.0f * r;
    unsigned char green = 255.0f * g;
    unsigned char blue = 255.0f * b;
    unsigned char alpha = 255;
    return (uchar4){red, green, blue, alpha};
}

__device__ uchar4 GetColor(int n, int max, int color)
{
    if (color == 0)
    {
        double t = (double)n / (double)max;

        unsigned int r = (int)(9 * (1 - t) * t * t * t * 255);
        unsigned int g = (int)(15 * (1 - t) * (1 - t) * t * t * 255);
        unsigned int b = (int)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

        return make_uchar4(r, g, b, 255);
    }
    else if (color == 1)
    {
        int N = 256;
        int N3 = N * N * N;
        double t = (double)n / (double)max;
        n = (int)(t * (double)N3);
        int b = n / (N * N);
        int nn = n - b * N * N;
        int r = nn / N;
        int g = nn - r * N;
        return make_uchar4(r, g, b, 255);
    }
    else if (color == 2)
    {
        float mmm = ((float)n / (float)max);
        return convert_one_pixel_to_rgb(mmm * 250, 1.0, 1.0);
    }
    else if (color == 3)
    {
        //float z = n%10*255/9;
        float z = ((n % 20) / 20.f) * 255;

        return convert_one_pixel_to_rgb(z, 1.0, 1.0);
    }
    else if (color == 4)
    {
        float z = n % 2 * 255;
        return make_uchar4(z, z, z, 255);
    }
}
__global__ void RenderKernel(uchar4 *dst, double SXSize, double SYSize, double FXSize, double FYSize, double FXMin, double FYMin, int iter_max, int smooth,double xpixelfact,double ypixelfact)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x);

    if (index < SXSize * SYSize)
    {

        int xc = index % (int)(SXSize);
        int yc = (int)((index - xc) / SXSize);

        unsigned int n = 0;
        double zx, zy, zx2, zy2;
        zx = zy = zx2 = zy2 = 0;
        double x = (double)xc * xpixelfact + FXMin; // x1
        double y = (double)yc * ypixelfact + FYMin; // x2

        for (; n < iter_max && zx2 + zy2 < 4; n++)
        {
            zy = 2 * zx * zy + y;
            zx = zx2 - zy2 + x;
            zx2 = zx * zx;
            zy2 = zy * zy;
        }
        
        dst[index] = GetColor(n, iter_max, smooth);        
    }
}

uchar4 *g_dstBuffer = NULL;
size_t g_BufferSize = 0;

void Render(cudaGraphicsResource_t &dst, Dimention<double> &screen, Dimention<double> &fract, int iter_max, int smooth)
{
    cudaGraphicsResource_t resources[1] = {dst};
    cudaGraphicsMapResources(1, resources);
    cudaArray *dstArray;
    cudaGraphicsSubResourceGetMappedArray(&dstArray, dst, 0, 0);

    size_t bufferSize = screen.width() * screen.height() * sizeof(uchar4);
    if (g_BufferSize != bufferSize)
    {
        if (g_dstBuffer != NULL)
        {
            cudaFree(g_dstBuffer);
        }
        g_BufferSize = bufferSize;
        cudaMalloc(&g_dstBuffer, g_BufferSize);
    }

    size_t blocksW = (size_t)ceilf(screen.width() / (float)BLOCK_SIZE);
    size_t blocksH = (size_t)ceilf(screen.height() / (float)BLOCK_SIZE);
    int n = screen.width() * screen.height();

    double xpixelfact =  1.f/(double)screen.width() * fract.width();
    double ypixelfact = 1.f/(double)screen.height() * fract.height();
    RenderKernel<<<n / BLOCK_SIZE, BLOCK_SIZE>>>(g_dstBuffer, screen.width(), screen.height(), fract.width(), fract.height(), fract.x_min(), fract.y_min(), iter_max, smooth,xpixelfact,ypixelfact);

    //cudaMemcpyToArray(dstArray, 0, 0, g_dstBuffer, bufferSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(dstArray, 0, 0, g_dstBuffer, screen.width() * sizeof(uchar4), screen.width() * sizeof(uchar4), screen.height(), cudaMemcpyDeviceToDevice);
    //TODO how to replace with cudaMemcpy2DToArray() ?;
    cudaGraphicsUnmapResources(1, resources);
}

__global__ void RenderKernelA(uchar4 *dst, double SXSize, double SYSize, double FXSize, double FYSize, double FXMin, double FYMin, int iter_max, int smooth, double radius, unsigned int size, thrust::complex<double> *xxx)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x);

    if (index < SXSize * SYSize)
    {

        int xc = index % (int)(SXSize);
        int yc = (int)((index - xc) / SXSize);

        double x = (double)xc / (double)SXSize * FXSize + FXMin; // x1
        double y = (double)yc / (double)SYSize * FYSize + FYMin; // x2

        int window_radius = (SXSize < SYSize) ? SXSize : SYSize;
        // find the complex number at the center of this pixel
        //thrust::complex<double> d0(radius * (2.f * xc - (double)SXSize) / (double)window_radius,
        // -radius * (2.f * yc - (double)SYSize) / (double)window_radius);

        thrust::complex<double> d0(x,y);

        int iter = 0;

        double zn_size;
        // run the iteration loop
        thrust::complex<double> dn = d0;
        do
        {
            dn *= xxx[iter] + dn;
            dn += d0;
            ++iter;
            zn_size = thrust::norm(xxx[iter] * thrust::complex<double>(0.5, 0.0) + dn);

            // use bailout radius of 256 for smooth coloring.
        } while (zn_size < 256 && iter < size);

        dst[index] = GetColor(iter, iter_max, smooth);
    }
}

void RenderA(cudaGraphicsResource_t &dst, Dimention<double> &screen, Dimention<double> &fract, int iter_max, int smooth, double radius, unsigned int size, thrust::complex<double> *x)
{
    cudaGraphicsResource_t resources[1] = {dst};
    cudaGraphicsMapResources(1, resources);
    cudaArray *dstArray;
    cudaGraphicsSubResourceGetMappedArray(&dstArray, dst, 0, 0);
    thrust::complex<double> *xgpu;
    cudaMalloc(&xgpu, size * sizeof(thrust::complex<double>));
    cudaMemcpy(x, xgpu, size * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

    size_t bufferSize = screen.width() * screen.height() * sizeof(uchar4);
    if (g_BufferSize != bufferSize)
    {
        if (g_dstBuffer != NULL)
        {
            cudaFree(g_dstBuffer);
        }
        g_BufferSize = bufferSize;
        cudaMalloc(&g_dstBuffer, g_BufferSize);
    }

    size_t blocksW = (size_t)ceilf(screen.width() / (float)BLOCK_SIZE);
    size_t blocksH = (size_t)ceilf(screen.height() / (float)BLOCK_SIZE);
    int n = screen.width() * screen.height();
    RenderKernelA<<<n / BLOCK_SIZE, BLOCK_SIZE>>>(g_dstBuffer, screen.width(), screen.height(), fract.width(), fract.height(), fract.x_min(), fract.y_min(), iter_max, smooth, radius, size, xgpu);

    cudaMemcpy2DToArray(dstArray, 0, 0, g_dstBuffer, screen.width() * sizeof(uchar4), screen.width() * sizeof(uchar4), screen.height(), cudaMemcpyDeviceToDevice);
    //TODO how to replace with cudaMemcpy2DToArray() ?;
    cudaGraphicsUnmapResources(1, resources);
    cudaFree(xgpu);
}