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

template<class Optimizer,class T=DTYPE>
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
  ANN ( const arma::Mat<T>& inputs,
        const arma::Mat<T>& labels,
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
  ANN ( const arma::Mat<T>& inputs,
        const arma::Mat<T>& labels,
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
  void Train ( const arma::Mat<T>& inputs,
               const arma::Mat<T>& labels );
  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void TrainEarlyStop ( const arma::Mat<T>& inputs,
                        const arma::Mat<T>& labels,
                        const size_t& patience );
  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Predict ( const arma::Mat<T>& inputs,
                 arma::Mat<T>& labels ) ;

  /**
   * Return/Modify the parameters of the network 
   */
  const arma::Mat<T>& Parameters (  ) const ;

  arma::Mat<T>& Parameters (  ) ;

  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  T ComputeError ( const arma::Mat<T>& points,
                   const arma::Mat<T>& responses );
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

  mlpack::FFN<mlpack::MeanSquaredErrorType<arma::Mat<DTYPE>>,
              mlpack::HeInitialization,
              arma::Mat<DTYPE>> model_;
};

} // namespace regression
} // namespace algo

#include "nn_impl.h"

#endif

