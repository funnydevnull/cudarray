#include "cudarray/common.hpp"
#include "cudarray/nnet/one_hot.hpp"

namespace cudarray {

template <typename T>
__global__ void kernel_one_hot_encode(const int *labels, int n_classes, int n,
                                      T *out) {
  CUDA_GRID_STRIDE_LOOP(idx, n*n_classes) {
    int class_idx = idx % n_classes;
    int label_idx = idx / n_classes;
    out[idx] = labels[label_idx] == class_idx ? 1.0 : 0.0;
  }
}

template <typename T>
void one_hot_encode(const int *labels, int n_classes, int n, T *out) {
  kernel_one_hot_encode<<<cuda_blocks(n*n_classes), kNumBlockThreads>>>(
      labels, n_classes, n, out);
  CUDA_KERNEL_CHECK;
}

template void one_hot_encode(const int *labels, int n_classes, int n,
                             float *out);

// copy rows

template <typename T>
__global__ void kernel_copy_rows(const int *rowids, int numrows, int rowsize,
                                      T *from_mat, T *to_mat) {
  CUDA_GRID_STRIDE_LOOP(idx, numrows*rowsize) {
	int row = idx / rowsize;
	int column = idx % rowsize;
	int from_row = rowids[row];
    to_mat[idx] = from_mat[from_row * rowsize + column];
  }
}

template <typename T>
__global__ void kernel_copy_rows_mapto(const int *rowids, int numrows, int rowsize,
                                      T *from_mat, T *to_mat) {
  CUDA_GRID_STRIDE_LOOP(idx, numrows*rowsize) {
	int row = idx / rowsize;
	int column = idx % rowsize;
	int from_row = rowids[row];
    to_mat[from_row * rowsize + column] = from_mat[idx];
  }
}

template <typename T>
void copy_rows(const int *rowids, int numrows, int rowsize, T *from_mat, T *to_mat, 
		int mapfrom) {
  if (mapfrom == 1) {
	kernel_copy_rows<<<cuda_blocks(numrows*rowsize), kNumBlockThreads>>>(
		rowids, numrows, rowsize, from_mat, to_mat);
	CUDA_KERNEL_CHECK;
  }
  else {
	kernel_copy_rows_mapto<<<cuda_blocks(numrows*rowsize), kNumBlockThreads>>>(
		rowids, numrows, rowsize, from_mat, to_mat);
	CUDA_KERNEL_CHECK;
  }
}

template void copy_rows(const int *rowids, int numrows, int rowsize, float *from_mat,
                             float *to_mat, int mapfrom);

// copy sum rows


template <typename T>
__global__ void kernel_copy_sum_rows(const int *rowids, T *coefficients, int
		numsums, int numrows, int rowsize, T *from_mat, T *to_mat, float constant, 
		float var) {
  CUDA_GRID_STRIDE_LOOP(idx, numrows*rowsize) {
	int row = idx / rowsize;
	int column = idx % rowsize;
	to_mat[idx]=0; 
	int from_row=0;
	if (coefficients != NULL ){
		for (int j=0; j < numsums; j++) {
			from_row = rowids[row * numsums + j];
			to_mat[idx] += coefficients[row * numsums + j] * 
							from_mat[from_row * rowsize + column];
		}
	}
	else {
		for (int j=0; j < numsums; j++) {
			from_row = rowids[row * numsums + j];
			to_mat[idx] +=  constant * pow(var, (float)j) * from_mat[from_row *
				rowsize + column];
		}
	}
  }
}

template <typename T>
__global__ void kernel_copy_sum_rows_mapto(const int *rowids, T *coefficients, int
		numsums, int numrows, int rowsize, T *from_mat, T *to_mat, float constant, 
		float var) {
  CUDA_GRID_STRIDE_LOOP(idx, numrows*rowsize) {
	int row = idx / rowsize;
	int column = idx % rowsize;
	//to_mat[idx]=0; 
	int from_row=0;
	if (coefficients != NULL ){
		for (int j=0; j < numsums; j++) {
			from_row = rowids[row * numsums + j];
			// we have to use an atomic add here because different threads are
			// writnig to the same memory location
			atomicAdd(&to_mat[from_row * rowsize + column], 
					coefficients[row * numsums + j] * from_mat[idx]);
		}
	}
	else {
		for (int j=0; j < numsums; j++) {
			from_row = rowids[row * numsums + j];
			// we have to use an atomic add here because different threads are
			// writnig to the same memory location
			atomicAdd(&to_mat[from_row * rowsize + column], constant *
					pow(var,(float)j)  *from_mat[idx]);
		}
	}
  }
}

template <typename T>
void copy_sum_rows(const int *rowids, int numsums, int numrows, int rowsize, 
		T *from_mat, T *to_mat, int mapfrom, T *coefficients, float constant, 
		float var) {
  if (mapfrom == 1) {
	kernel_copy_sum_rows<<<cuda_blocks(numrows*rowsize), kNumBlockThreads>>>(
		rowids, coefficients, numsums, numrows, rowsize, from_mat, to_mat,
		constant, var);
	CUDA_KERNEL_CHECK;
  }
  else {
	kernel_copy_sum_rows_mapto<<<cuda_blocks(numrows*rowsize), kNumBlockThreads>>>(
		rowids, coefficients, numsums, numrows, rowsize, from_mat, to_mat,
		constant, var);
	CUDA_KERNEL_CHECK;
  }
}

template void copy_sum_rows(const int *rowids, int numsums, int numrows, int
		rowsize, float *from_mat, float *to_mat, int mapfrom, 
		float *coefficients, float constant, float var);

}
