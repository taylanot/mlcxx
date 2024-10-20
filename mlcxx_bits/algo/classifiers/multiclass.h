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
class OnevAll
{
public:
  /**
   * Non-working model 
   */
  OnevAll (  ) = default;

  /**
   * @param args    : parameters for the model
   */
  template<class... Args>
  OnevAll ( const size_t& num_class, const Args&... args );


  /**
   * @param inputs  : X
   * @param labels  : y
   * @param args    : parameters for the model
   */
  template<class... Args>
  OnevAll ( const arma::Mat<T>& inputs,
            const arma::Row<size_t>& labels,
            const size_t& num_class,
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
   * @param inputs  : X
   * @param labels  : y
   * @param args    : parameters for the model
   */
  template<class... Args>
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels );
  /**
   * @param inputs  : X*
   * @param preds   : y*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& preds );
  /**
   * @param inputs  : X*
   * @param preds   : y*
   * @param probs   : p*
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

  std::vector<MODEL> models_;

private:
  size_t nclass_;
  size_t unclass_;
  arma::Row<size_t> unq_;
  bool oneclass_ = false;
};

} // namespace classification
} // namespace algo

#include "multiclass_impl.h"

#endif
