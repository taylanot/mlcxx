/**
 * @file convert.h
 * @author Ozgur Taylan Turan
 *
 * Conversion between armadillo matrix and libtorch tensors by means of getting
 * the data from the pointers
 *
 * TODO: Add jem and boost version cross conversions
 */

#ifndef CONVERT_H
#define CONVERT_H

//// standard
//#include <string>
//// armadillo 
//#include <armadillo>
//// torch 
////#include <ATen/ATen.h>
////#include <torch/torch.h>
//// jem
//#include <jem/base/array/Array.h>
//#include <jem/base/String.h>
//// boost 
//#include <boost/assert.hpp>
//#include <boost/numeric/ublas/matrix.hpp>

namespace utils {
  
////=============================================================================
//// to_Mat (Need to Test)
////=============================================================================
//template<class T>
//arma::Mat<T> to_dMat( at::Tensor data )
//{
//  T* dataptr = data.data_ptr<T>();
//  arma::mat data_converted(dataptr,
//                                  data.sizes()[1],
//                                  data.sizes()[0],false,false);
//  return arma::trans(data_converted);
//}
////=============================================================================
//// to_dmat
////=============================================================================
//
//arma::Mat<double> to_dMat( at::Tensor data )
//{
//  BOOST_ASSERT_MSG( data.scalar_type() == at::kDouble, 
//      "Data conversion is not possible due to type mismatch!" );
//  double* dataptr = data.data_ptr<double>();
//  arma::mat data_converted(dataptr,
//                                  data.sizes()[1],
//                                  data.sizes()[0],false,false);
//  return arma::trans(data_converted);
//}
//
////=============================================================================
//// to_fmat
////=============================================================================
//
//arma::Mat<float> to_fMat( at::Tensor data )
//{
//  BOOST_ASSERT_MSG( data.scalar_type() == at::kFloat, 
//      "Data conversion is not possible due to type mismatch!" );
//  float* dataptr = data.data_ptr<float>();
//  arma::fmat data_converted(dataptr,
//                                  data.sizes()[1],
//                                  data.sizes()[0],false,false);
//  return arma::trans(data_converted);
//}
//
//
////=============================================================================
//// to_imat
////=============================================================================
//
//arma::Mat<int> to_iMat( at::Tensor data )
//{
//  BOOST_ASSERT_MSG( data.scalar_type() == at::kInt, 
//      "Data conversion is not possible due to type mismatch!" );
//  int* dataptr = data.data_ptr<int>();
//  arma::Mat<int> data_converted(dataptr,
//                                  data.sizes()[1],
//                                  data.sizes()[0],false,false);
//  return arma::trans(data_converted);
//}

////=============================================================================
//// to_Tensor
////=============================================================================
//
//torch::Tensor to_Tensor( arma::Mat<double> data )
//{
//  double* dataptr = data.memptr();
//  torch::Tensor converted_data = torch::from_blob(dataptr, 
//                                                  {int(data.n_cols),
//                                                  int(data.n_rows)},
//                                                  torch::TensorOptions()
//                                                  .dtype(torch::kDouble));
//  return converted_data.transpose(0,1);
//}
//
//torch::Tensor to_Tensor( arma::Mat<float> data )
//{
//  float* dataptr = data.memptr();
//  torch::Tensor converted_data = torch::from_blob(dataptr,
//                                                  {int(data.n_cols),
//                                                  int(data.n_rows)},
//                                                  torch::TensorOptions()
//                                                  .dtype(torch::kFloat));
//  return converted_data.transpose(0,1);
//}
//
//torch::Tensor to_Tensor( arma::Mat<int> data )
//{
//  int* dataptr = data.memptr();
//  torch::Tensor converted_data = torch::from_blob(dataptr,
//                                                  {int(data.n_cols),
//                                                  int(data.n_rows)},
//                                                  torch::TensorOptions()
//                                                  .dtype(torch::kInt));
//  return converted_data.transpose(0,1);
//}

//=============================================================================
// to_string
//=============================================================================


std::string to_string ( const jem::String& data )
{
  const char* dataptr = data.addr();
  return std::string(dataptr, data.size());
}

//=============================================================================
// to_char
//=============================================================================


const char* to_char ( const jem::String& data )
{
  const char* dataptr = data.addr();
  return dataptr;
}

} // namespace utils

#endif
