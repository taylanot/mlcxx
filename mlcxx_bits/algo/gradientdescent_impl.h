/**
 * @file gradientdescent_impl.h
 * @author Ozgur Taylan Turan
 *
 * Simple Gradient Descent Models
 *
 */
#ifndef GRADIENTDESCENT_IMPL_H
#define GRADIENTDESCENT_IMPL_H

// mlpack
#include <mlpack/core.hpp>
// local
#include "kernelridge.h"
#include "utils/covmat.h"

using namespace mlpack;
using namespace algo::regression;

///////////////////////////////////////////////////////////////////////////////
// Linear Regression
///////////////////////////////////////////////////////////////////////////////

GDLinear::GDLinear( const arma::mat& inputs,
                    const arma::rowvec& labels,
                    const bool bias,
                    const double lambda,
                    const double lr,
                    const size_t t ) :
          lambda_(lambda), bias_(bias), lr_(lr), t_(t)
{
  Train(inputs, labels);
}


void GDLinear::Train( const arma::mat& inputs,
                      const arma::rowvec& labels )
{
  // randomly initialize parameters  
  arma::mat X = inputs;
  if (bias_)
  {
    parameters_ = arma::vec(2, arma::fill::randn);
    X.insert_rows(inputs.n_rows, arma::ones(inputs.n_cols));
  }
  else
    parameters_ = arma::vec(1, arma::fill::randn);

  arma::rowvec preds;
  // update the parameters in a loop with gradients
  for ( size_t i=0; i<t_; i++ )
  {
    Predict(inputs, preds);
    dparameters_ = (-2./X.n_rows) * ( X * (labels - preds) );
    parameters_ += lr_ * dparameters_;
  }
}

void GDLinear::Predict( const arma::mat& inputs,
                        arma::rowvec& labels ) const
{
  arma::mat X = inputs;
  if (bias_)
    X.insert_rows(inputs.n_rows, arma::ones(inputs.n_cols));
  labels = inputs * X;
}

double GDLinear::ComputeError( const arma::mat& inputs,
                               const arma::rowvec& labels ) const
{
  arma::rowvec temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const double cost = arma::dot(temp, temp) / n_points;

  return cost;
}

#endif 
