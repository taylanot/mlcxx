/**
 * @file functionalsmoothing.h
 * @author Ozgur Taylan Turan
 *
 * Smoothing for functional data
 *
 *
 *
 */
#ifndef FUNCTIONALSMOOTHING_H
#define FUNCTIONALSMOOTHING_H

namespace utils {
namespace functional {

template<class T>
arma::mat kernelsmoothing ( const arma::mat& inputs,
                            const arma::mat& labels,
                            const arma::mat& pred_inputs,
                            const arma::vec& bandwidths,
                            const double valid)
{
  size_t N = pred_inputs.n_cols;
  size_t M = labels.n_rows;

  arma::mat predictions = arma::zeros(M, N);

  for(size_t i=0; i<M; i++)
  {
    arma::rowvec label = labels.row(i);
    arma::rowvec temp;

    mlpack::hpt::HyperParameterTuner<algo::regression::Kernel<T>,
                                     mlpack::cv::MSE,
                                     mlpack::cv::SimpleCV>
                                     hpt(valid, inputs, label);


    double bandwidth;
    std::tie(bandwidth) = hpt.Optimize(bandwidths);

    algo::regression::Kernel<T> smoother(inputs,label,bandwidth);

    smoother.Predict(pred_inputs,temp);
    predictions.row(i) = temp;
  }  
  return predictions;
}

template<class T>
arma::mat kernelsmoothing ( const arma::mat& inputs,
                            const arma::mat& labels,
                            const arma::mat& pred_inputs,
                            const double& bandwidth    )
{
  size_t N = pred_inputs.n_cols;
  size_t M = labels.n_rows;

  arma::mat predictions = arma::zeros(M, N);

  for(size_t i=0; i<M; i++)
  {
    arma::rowvec label = labels.row(i);
    arma::rowvec temp;

    algo::regression::Kernel<T> smoother(inputs,label,bandwidth);

    smoother.Predict(pred_inputs,temp);
    predictions.row(i) = temp;
  }  
  return predictions;
}
} // functional
} // utils
#endif 
