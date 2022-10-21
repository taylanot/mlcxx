/**
 * @file convert.h
 * @author Ozgur Taylan Turan
 *
 * Conversion between armadillo matrix and libtorch tensors by means of getting
 * the data from the pointers
 *
 * TODO: Add jem and boost version cross conversions
 *
 */

#ifndef CONVERT_H
#define CONVERT_H

// standard
#include <string>
// armadillo 
#include <armadillo>
// torch 
#include <ATen/ATen.h>
#include <torch/torch.h>
// jem
#include <jem/base/array/Array.h>
#include <jem/base/String.h>
// boost 
#include <boost/numeric/ublas/matrix.hpp>
namespace utils {

torch::Tensor to_Tensor ( arma::Mat<double> data );
torch::Tensor to_Tensor ( arma::Mat<float> data );
torch::Tensor to_Tensor ( arma::Mat<int> data );

arma::Mat<double> to_dMat ( at::Tensor data );
arma::Mat<float>  to_fMat ( at::Tensor data );
arma::Mat<int>    to_iMat ( at::Tensor data );

std::string to_string ( jem::String data );

} //namespace utils

#endif
