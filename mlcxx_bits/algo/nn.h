/**
 * @file nn.h
 * @author Ozgur Taylan Turan
 *
 * Simple neural network wrapper for using learning curve generation and 
 * hyper-parameter tuning.
 *
 * TODO: 
 *  Pre-trained model loading can be done.
 *
 *
 */
#ifndef NN_H
#define NN_H

namespace algo {

template<class NET,
         class OPT=ens::StandardSGD,
         class MET=mlpack::SquaredEuclideanDistance,
         class O=DTYPE>
class ANN
{
public:
  /**
   * Non-working model 
   */
  ANN ( ) { };

  /**
   * @param network : pointer to the network object
   * @param args    : optimizer arguments in the given order
   */
  template<class... OptArgs>
  ANN ( NET* network, bool early = false, const OptArgs&... args );

  /**
   * @param inputs  : input data, X
   * @param labels  : labels of the input, y
   * @param network : pointer to the network object
   * @param args    : optimizer arguments in the given order
   */
  template<class... OptArgs>
  ANN ( const arma::Mat<O>& inputs,
        const arma::Mat<O>& labels,
        NET* network, bool early = false, const OptArgs&... args ) ;
  /**
   * @param inputs  : input data, X
   * @param labels  : labels of the input, y
   */
  void Train( const arma::Mat<O>& inputs, const arma::Mat<O>& labels );

   /**
   * @param inputs  : input data, X
   * @param preds   : prediction of labels of the input, \hat{y}
   */ 
  void Predict( const arma::Mat<O>& inputs, arma::Mat<O>& preds );

  /**
   * @param inputs  : input data, X
   * @param preds   : prediction of labels of the input, \hat{y}
   */ 
  void Classify( const arma::Mat<O>& inputs, arma::Row<size_t>& preds );


  /**
   * @param inputs  : input data, X
   * @param labels  : labels of the input, y
   */ 
  O ComputeError( const arma::Mat<O>& inputs, const arma::Mat<O>& labels );


private:
  // A unique pointer for the optimizer. This is needed because of design
  // choices in ensmallen optimizers based on SGD.
  std::unique_ptr<OPT> opt_; 
  // A predefined network pointer. We have to define the network outside with
  // this construction. This gives more freedom on how you want to create your
  // network. Moreover, it gives you the flexibility to 
  NET* network_;

  bool early_;

};

} // namespace algo

#include "nn_impl.h"

#endif


