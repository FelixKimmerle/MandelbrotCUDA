#include "Dimentions.h"
#include <cuda_gl_interop.h>
#include <thrust/complex.h>
void Render(cudaGraphicsResource_t &dst, Dimention<double> &screen, Dimention<double> &fract, int iter_max, int smooth);
void RenderA(cudaGraphicsResource_t &dst, Dimention<double> &screen, Dimention<double> &fract, int iter_max, int smooth,double radius,unsigned int size,thrust::complex<double> *x);