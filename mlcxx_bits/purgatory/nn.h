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
template<class Optimizer, class T = DTYPE>
class ANN 
{
public:
    ANN() = default;

    template<class... Args>
    ANN(const arma::Mat<T>& inputs, const arma::Mat<T>& labels, const Args&... args);

    template<class... Args>
    ANN(const arma::Mat<T>& inputs, const arma::Mat<T>& labels, const size_t& archtype = 0, const size_t& nonlintype = 0, const bool& early = false, const Args&... args);

    /* // Copy constructor */
    /* ANN(const ANN& other); */

    /* // Move constructor */
    /* ANN(ANN&& other) noexcept; */

    /* // Copy assignment operator */
    /* ANN& operator=(const ANN& other); */

    /* // Move assignment operator */
    /* ANN& operator=(ANN&& other) noexcept; */

    void Train(const arma::Mat<T>& inputs, const arma::Mat<T>& labels);

    void Predict(const arma::Mat<T>& inputs, arma::Mat<T>& labels);

    const arma::Mat<T>& Parameters() const;

    arma::Mat<T>& Parameters();

private:
    arma::Row<size_t> layer_info_;
    size_t nonlintype_;
    size_t archtype_;
    bool early_;
    size_t ptc_ = 10;
    std::unique_ptr<Optimizer> opt_;
    double potentialsplitratio_ = 0.2;
    mlpack::FFN<mlpack::MeanSquaredErrorType<arma::Mat<DTYPE>>, mlpack::HeInitialization, arma::Mat<DTYPE>> model_;
};

/*template<class Optimizer,class T=DTYPE> */
/*class ANN */ 
/*{ */
/*  public: */
/*  ANN ( ) {}; */
/*  /1** */
/*   * @param inputs      :  X */
/*   * @param labels      :  y */
/*   * @param archtype    :  preset architecture type */ 
/*   * @param nonlintype  :  preset nonlinearity type */ 
/*   * @param args        :  arguments for optimizer */
/*   *1/ */

/*  template<class... Args> */
/*  ANN ( const arma::Mat<T>& inputs, */
/*        const arma::Mat<T>& labels, */
/*        const Args&... args ); */

/*  template<class... Args> */
/*  ANN ( const arma::Mat<T>& inputs, */
/*        const arma::Mat<T>& labels, */
/*        const size_t& archtype = 0 , */
/*        const size_t& nonlintype = 0, */
/*        const bool& early = false, */
/*        const Args&... args ); */

/*   // Copy constructor */
/*  /1* ANN(const ANN& other); *1/ */
/*  /1* // Copy assignment operator *1/ */
/*  /1* ANN& operator=(const ANN& other); *1/ */

/*// Copy constructor */
/*    ANN(const ANN& other); */

/*    // Move constructor */
/*    ANN(ANN&& other) noexcept = default; */

/*    ANN& operator=(const ANN& other) { */
/*    // Implement the copy assignment logic here */
/*    if (this != &other) { */
/*        // Perform deep copy of other's members */
/*    } */
/*    return *this; */
/*} */
/*  /1** */
/*   * @param inputs  : X */
/*   * @param labels  : y */
/*   *1/ */
/*  void Train ( const arma::Mat<T>& inputs, */
/*               const arma::Mat<T>& labels ); */
/*  /1** */
/*   * @param inputs  : X* */
/*   * @param labels  : y* */
/*   *1/ */
/*  void Predict ( const arma::Mat<T>& inputs, */
/*                 arma::Mat<T>& labels ) ; */

/*  /1** */
/*   * Return/Modify the parameters of the network */ 
/*   *1/ */
/*  const arma::Mat<T>& Parameters (  ) const ; */

/*  arma::Mat<T>& Parameters (  ) ; */

/*  private: */
/*  arma::Row<size_t> layer_info_; */
/*  size_t nonlintype_; */ 
/*  size_t archtype_; */ 
/*  bool early_; */
/*  size_t ptc_ = 10; */


/*  std::unique_ptr<Optimizer> opt_; */

/*  double potetialsplitratio_ = 0.2; */ 

/*  mlpack::FFN<mlpack::MeanSquaredErrorType<arma::Mat<DTYPE>>, */
/*              mlpack::HeInitialization, */
/*              arma::Mat<DTYPE>> model_; */
/*}; */

} // namespace regression
} // namespace algo

#include "nn_impl.h"

#endif

