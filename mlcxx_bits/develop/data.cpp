/**
 * @file data.cpp
 * @author Ozgur Taylan Turan
 *
 * Combining my dataset implementations in a nice way... 
 *
 */

#include <headers.h>

// classification set
/* using DATASET = data::oml::Dataset<size_t>; */
using DATASET = data::Dataset<arma::Row<size_t>>;

int main ( ) 
{
  size_t N = 10;
  DATASET data(size_t(2));
  data.Banana(N);
  data.Dipping(N);
  /* arma::Row<DTYPE> a = {1,2}; */
  /* arma::Row<DTYPE> b = {1,1}; */
  /* data.Gaussian(100); */
  /* data.Generate("Linear"); */
  PRINT(data.inputs_);
  PRINT(data.labels_);

}
