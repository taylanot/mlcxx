/**
 * @file nn.h
 * @author Ozgur Taylan Turan
 *
 * Simple neural network wrapper for using learning curve generation and 
 * hyper-parameter tuning.
 *
 * TODO: 
 *  Pre-trained model loading.
 *
 *
 */
#ifndef NN_H
#define NN_H

namespace algo {

template<class NET,
         class OPT=ens::StandardSGD,
         class MET=mlpack::MeanSquaredError,
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
  /* template<class... OptArgs> */
  /* ANN ( NET* network, bool early = false, const OptArgs&... args ); */

  /**
   * @param inputs  : input data, X
   * @param labels  : labels of the input, y
  */  
  ANN ( const arma::Mat<O>& inputs,
        const arma::Mat<O>& labels ) ;

  /**
   * @param inputs  : input data, X
   * @param labels  : labels of the input, y
   * @param network : network object
   */
  
  ANN ( const arma::Mat<O>& inputs,
        const arma::Mat<O>& labels,
        const NET network   ) ;

  template<class... OptArgs>
  ANN ( const arma::Mat<O>& inputs,
        const arma::Mat<O>& labels,
        const NET network, bool early = false, const OptArgs&... args ) ;

  template<class... OptArgs>
  ANN ( const arma::Mat<O>& inputs,
        const arma::Row<size_t>& labels,
        const NET network, bool early = false, const OptArgs&... args ) ;
  /**
   * @param inputs  : input data, X
   * @param labels  : labels of the input, y
   */
  void Train( const arma::Mat<O>& inputs, const arma::Mat<O>& labels );

  template<class... OptArgs>
  void Train( const arma::Mat<O>& inputs, const arma::Mat<O>& labels,
              const NET network, bool early = false, const OptArgs&... args  );

  template<class... OptArgs>
  void Train( const arma::Mat<O>& inputs, const arma::Row<size_t>& labels,
              const NET network, bool early = false, const OptArgs&... args  );

  void Train( const arma::Mat<O>& inputs, const arma::Row<size_t>& labels );

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

  arma::Mat<O> Parameters( );

  template<class Archive>
  void serialize(Archive& ar) 
  { 
    ar( CEREAL_NVP(network_),
        CEREAL_NVP(early_),
        CEREAL_NVP(ulab_) ); 
  }


private:
  /**
   * @param labels : labels to be encoded
   * @param map    : expected labels
   */
  arma::Mat<O> _OneHotEncode ( const arma::Row<size_t>& labels,
                               const arma::Row<size_t>& ulabels );

  /**
   * @param labels : labels to be decoded 
   * @param map    : expected labels
   */
  arma::Row<size_t> _OneHotDecode ( const arma::Mat<O>& labels,
                                    const arma::Row<size_t>& ulabels);

  // A unique pointer for the optimizer. This is needed because of design
  // choices in ensmallen optimizers based on SGD.
  std::unique_ptr<OPT> opt_; // optimizer pointer 
                             
  // A predefined network pointer. We have to define the network outside with
  // this construction. This gives more freedom on how you want to create your
  // network. Moreover, it gives you the flexibility to 
  /* NET* network_; // network pointer */
  NET network_; // network pointer

  bool early_; // Early stopping flag
               
  // ONLY VALID FOR CLASSIFICATION PROBLEMS
  arma::Row<size_t> ulab_; // Just a container for the potential unique labels

};

} // namespace algo

#include "nn_impl.h"

#endif
