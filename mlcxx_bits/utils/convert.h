/**
 * @file utils.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef CONVERT_H
#define CONVERT_H

namespace utils {

template<class T>
void CSC( const arma::Mat<T>& mat, std::vector<DTYPE>& values,
          std::vector<int>& row_indices, std::vector<int>& col_start) 
{
  int n_rows = mat.n_rows;
  int n_cols = mat.n_cols;

  // Clear the output vectors
  values.clear();
  row_indices.clear();
  col_start.clear();

  col_start.push_back(0); // First column starts at index 0

  // Iterate over each column
  for (int col = 0; col < n_cols; ++col)
  {
    // Iterate over each row in the current column
    for (int row = 0; row < n_rows; ++row)
    {
      double value = mat(row, col);
      if (value != 0.0)
      {
        values.push_back(value);
        row_indices.push_back(row);
      }
    }
    col_start.push_back(values.size());
  }
}

} // namespace utils
#endif
