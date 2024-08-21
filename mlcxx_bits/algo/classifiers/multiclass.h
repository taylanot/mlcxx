/**
 * @file multiclass.h
 * @author Ozgur Taylan Turan
 *
 * MultiClass Classifier for binary classifiers
 *
 */

#ifndef MULTICLASS_H
#define MULTICLASS_H

namespace algo {
namespace classification {

template<class MODEL, class T=DTYPE>
class MultiClass
{
public:
  /**
   * Non-working model 
   */
  MultiClass (  ) = default;

  /**
   * Non-working model 
   * @param type  : algorithm OnevsAll vs OnevsOne
   */
  MultiClass ( std::string type ) : type_(type) { }

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param type    :algorithm OnevsAll vs OnevsOne
   * @param args    : parameters for the model
   */
  template<class... Args>
  MultiClass ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels,
               const std::string type,
               const Args&... args );

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param args    : parameters for the model
   */
  template<class... Args>
  MultiClass ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels,
               const Args&... args );

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param args    : parameters for the model
   */
  template<class... Args>
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels,
               const Args&... args );
  /**
   * @param inputs  : X*
   * @param preds   : y*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& preds );
   /**
   * @param inputs  : X*
   * @param preds   : y*
   * @param probs   : ps
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& preds,
                  arma::Mat<T>& probs );
  /**
   * @param inputs  : X
   * @param labels  : y
   */
  T ComputeError ( const arma::Mat<T>& inputs,
                   const arma::Row<size_t>& labels );

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  T ComputeAccuracy ( const arma::Mat<T>& inputs,
                      const arma::Row<size_t>& labels );

private:

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param args    : parameters for the model
   */
  template<class... Args>
  void OneVsAll ( const arma::Mat<T>& inputs,
                  const arma::Row<size_t>& labels,
                  const Args&... args );


  std::vector<MODEL> models_;
  size_t nclass_;
  arma::Row<size_t> unq_;
  std::string type_;
  bool oneclass_ = false;
};
} // namespace classification
} // namespace algo

#include "multiclass_impl.h"

#endif
