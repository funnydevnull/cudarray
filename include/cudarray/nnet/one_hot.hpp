#ifndef ONE_HOT_HPP_
#define ONE_HOT_HPP_

namespace cudarray {

template <typename T>
void one_hot_encode(const int *labels, int n_classes, int n, T *out);

template <typename T>
void copy_rows(const int *rowids, int numrows, int rowsize, T* from_mat, T *to_mat,
		int mapfrom);

template <typename T>
void copy_sum_rows(const int *rowids, int numsums, int numrows, int rowsize, 
		T* from_mat, T *to_mat, int mapfrom, T* coefficients, 
		float constant, float var);

}

#endif  // ONE_HOT_HPP_
