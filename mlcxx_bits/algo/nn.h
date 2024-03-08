/**
 * @file nn.h
 * @author Ozgur Taylan Turan
 *
 * Simple Neural Network Wrapper
 *
 */
#ifndef NN_H
#define NN_H

namespace algo { 
namespace regression {



//=============================================================================
// ANN
//=============================================================================

template<class Optimizer>
class ANN 
{
  public:
  /**
   * @param inputs      :  X
   * @param labels      :  y
   * @param archtype    :  preset architecture type 
   * @param nonlintype  :  preset nonlinearity type 
   * @param args        :  arguments for optimizer
   */
  template<class... Args>
  ANN ( const arma::mat& inputs,
       const arma::mat& labels,
       const size_t& archtype,
       const size_t& nonlintype,
       const Args&... args );


  /**
   * @param inputs      :  X
   * @param labels      :  y
   * @param layer_info  :  hidden layer neurons
   * @param nonlintype  :  preset nonlinearity type 
   * @param args        :  arguments for optimizer
   */
  template<class... Args>
  ANN ( const arma::mat& inputs,
       const arma::mat& labels,
       const arma::Row<size_t>& layer_info,
       const size_t& nonlintype,
       const Args&... args );

  /**
   * @param layer_info  :  hidden layer neurons
   * @param nonlintype  :  preset nonlinearity type 
   * @param args        :  arguments for optimizer
   */
  template<class... Args>
  ANN ( const arma::Row<size_t>& layer_info,
       const size_t& nonlintype,
       const Args&... args );

  /**
   * Non-working model 
   */
  //NN(): layer_info_({0})  { }

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::mat& inputs,
               const arma::mat& labels );
  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void TrainEarlyStop ( const arma::mat& inputs,
                        const arma::mat& labels,
                        const size_t& patience );
  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Predict ( const arma::mat& inputs,
                 arma::mat& labels ) ;

  /**
   * Return/Modify the parameters of the network 
   */
  const arma::mat& Parameters (  ) const ;

  arma::mat& Parameters (  ) ;

  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  double ComputeError ( const arma::mat& points,
                        const arma::mat& responses );
  /**
   * Set the learning rate 
   * @param lr : learning rate
   */
  void StepSize ( const double& lr );

  /**
   * Set the batch size
   * @param bs : batch size
   */
  void BatchSize ( const size_t& bs );

  /**
   * Set the number of epochs
   * @param epochs : epochs
   */
  void MaxIterations ( const size_t& epochs );

  /**
   * Save the model_
   * @param filename  : name of the file (.bin) extension
   * @param name      : name of the model
   */
  void Save ( const std::string& filename );
  /**
   * Load the model_
   * @param name      : name of the model
   */
  void Load ( const std::string& filename );

  private:
  arma::Row<size_t> layer_info_;
  size_t nonlintype_; 
  size_t archtype_; 

  Optimizer opt_; 

  double potetialsplitratio_ = 0.2; 

  mlpack::FFN<mlpack::MeanSquaredError, mlpack::HeInitialization> model_;
};

//=============================================================================
// NN
//=============================================================================

template<class Optimizer>
class NN 
{
  public:

  /**
   * @param inputs      :  X
   * @param labels      :  y
   * @param layer_info  :  hidden layer neurons
   * @param args        :  arguments for optimizer
   */
  template<class... Args>
  NN ( const arma::mat& inputs,
       const arma::mat& labels,
       const arma::Row<size_t>& layer_info,
       const Args&... args );

  template<class... Args>
  NN ( const arma::mat& inputs,
       const arma::rowvec& labels,
       const arma::Row<size_t>& layer_info,
       const Args&... args );

  /**
   * @param layer_info  :  hidden layer neurons
   * @param args        :  arguments for optimizer
   */
  template<class... Args>
  NN ( const arma::Row<size_t>& layer_info,
       const Args&... args );

  /**
   * Non-working model 
   */
  //NN(): layer_info_({0})  { }

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::mat& inputs,
               const arma::mat& labels );
  /**
   * @param inputs  : X*
   * @param labels  : y*
   */

  void Predict ( const arma::mat& inputs,
                 arma::mat& labels ) ;
  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  double ComputeError ( const arma::mat& points,
                        const arma::mat& responses );
  /**
   * Set the learning rate 
   * @param lr : learning rate
   */
  void StepSize ( const double& lr );

  /**
   * Set the batch size
   * @param bs : batch size
   */
  void BatchSize ( const size_t& bs );

  /**
   * Set the number of epochs
   * @param epochs : epochs
   */
  void MaxIterations ( const size_t& epochs );

  /**
   * Save the model_
   * @param filename  : name of the file (.bin) extension
   * @param name      : name of the model
   */
  void Save ( const std::string& filename );
  /**
   * Load the model_
   * @param name      : name of the model
   */
  void Load ( const std::string& filename );

  private:
  arma::Row<size_t> layer_info_;
  Optimizer opt_; 

  mlpack::FFN<mlpack::MeanSquaredError, mlpack::HeInitialization> model_;
};

//namespace my 
//{
//
//template<typename MatType = arma::mat>
//class MeanSquaredErrorType
//{
// public:
//  MeanSquaredErrorType ( ) { }
//
//  /**
//   * Computes the mean squared error function.
//   *
//   * @param prediction Predictions used for evaluating the specified loss
//   *     function.
//   * @param target The target vector.
//   */
//  typename MatType::elem_type Forward(const MatType& prediction,
//                                      const MatType& target)
//  {
//  typename MatType::elem_type lossSum =
//      arma::accu(arma::square(prediction - target));
//
//  return lossSum / target.n_elem;
//  }
//  /**
//   * Ordinary feed backward pass of a neural network.
//   *
//   * @param prediction Predictions used for evaluating the specified loss
//   * function
//   * @param target The target vector.
//   * @param loss The calculated error.
//   */
//  void Backward(const MatType& prediction,
//                const MatType& target,
//                MatType& loss)
//  {
//    loss = 2 * (prediction - target);
//
//    loss = loss / target.n_elem;
//  }
//  /**
//   * Serialize the layer
//   */
//  template<typename Archive>
//  void serialize(Archive& ar, const uint32_t /* version */) { }
// private:
//
//};
//typedef MeanSquaredErrorType<arma::mat> MeanSquaredError;
//}
} // namespace regression
} // namespace algo

#include "nn_impl.h"

#endif

