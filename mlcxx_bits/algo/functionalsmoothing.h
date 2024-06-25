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

namespace algo {
namespace functional {

template<class KERNEL,class T=DTYPE>
arma::Mat<T> kernelsmoothing ( const arma::Mat<T>& inputs,
                            const arma::Mat<T>& labels,
                            const arma::Mat<T>& pred_inputs,
                            const arma::vec& bandwidths,
                            const double valid)
{
  size_t N = pred_inputs.n_cols;
  size_t M = labels.n_rows;

  arma::Mat<T> predictions = arma::zeros<arma::Mat<T>>(M, N);

  for(size_t i=0; i<M; i++)
  {
    arma::Row<T> label = labels.row(i);
    arma::Row<T> temp;

    mlpack::HyperParameterTuner<algo::regression::Kernel<KERNEL>,
                                     mlpack::MSE,
                                     mlpack::SimpleCV>
                                     hpt(valid, inputs, label);


    double bandwidth;
    std::tie(bandwidth) = hpt.Optimize(bandwidths);

    algo::regression::Kernel<KERNEL> smoother(inputs,label,bandwidth);

    smoother.Predict(pred_inputs,temp);
    predictions.row(i) = temp;
  }  
  return predictions;
}

template<class KERNEL,class T=DTYPE>
arma::Mat<T> kernelsmoothing ( const arma::Mat<T>& inputs,
                               const arma::Mat<T>& labels,
                               const arma::Mat<T>& pred_inputs,
                               const double& bandwidth    )
{
  size_t N = pred_inputs.n_cols;
  size_t M = labels.n_rows;

  arma::Mat<T> predictions = arma::zeros<arma::Mat<T>>(M, N);

  for(size_t i=0; i<M; i++)
  {
    arma::Row<T> label = labels.row(i);
    arma::Row<T> temp;

    algo::regression::Kernel<KERNEL> smoother(inputs,label,bandwidth);

    smoother.Predict(pred_inputs,temp);
    predictions.row(i) = temp;
  }  
  return predictions;
}

template<class KERNEL,class T=double>
arma::Mat<T> kernelsmoothing ( const arma::Mat<T>& inputs,
                               const arma::Mat<T>& labels,
                               const double& bandwidth    )

{
  return kernelsmoothing<KERNEL>( inputs,labels,inputs,bandwidth );

}

} // functional
} // utils
#endif 
