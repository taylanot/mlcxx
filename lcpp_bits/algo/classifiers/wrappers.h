/**
 * @file wrapper.h
 * @author Ozgur Taylan Turan
 *
 * This file is for the wrappers for some mlpack model for imporved use
 *
 */

#ifndef CLASS_WRAPPER
#define CLASS_WRAPPER

namespace algo::classification {
//-----------------------------------------------------------------------------
// Logistic Regression
//-----------------------------------------------------------------------------
template<class T=DTYPE>
class LogisticRegression
{
  public:

  LogisticRegression ( ) : lambda_(0.) { };

  /**
   * @param lambda  : regularization
   */
  LogisticRegression ( const size_t& num_class ) 
    : num_class_(num_class), lambda_(0.) { };

  LogisticRegression ( const size_t& num_class, const double& lambda ) 
    : num_class_(num_class), lambda_(lambda) { };

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param num_class : number of classes
   */
  LogisticRegression ( const arma::Mat<T>& inputs,
                       const arma::Row<size_t>& labels,
                       const size_t& num_class ) 
                        : num_class_(num_class), lambda_(0.)
  {
    Train(inputs,labels); 
  }
 
  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   * @param lambda    : regularization
   */
  LogisticRegression ( const arma::Mat<T>& inputs,
                       const arma::Row<size_t>& labels,
                       const size_t& num_class,
                       const double& lambda ) 
                        : num_class_(num_class), lambda_(lambda)
  {
    Train(inputs,labels); 
  }

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels ) 
  {
    ulab_ = arma::unique(labels);
    if (ulab_.n_elem == 1)
      return;
    else
    {
      if (ulab_.n_elem == 2)
      {
        model_ = mlpack::LogisticRegression(inputs.n_rows,lambda_);
        model_.Train(inputs,labels);
      }
      else
      {
        ova_ = OnevAll<mlpack::LogisticRegression<arma::Mat<T>>> 
                                (num_class_,inputs.n_rows,lambda_);
        ova_.Train(inputs, labels);
      }
    }
  }

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param num_class: number of classes
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels,
               const size_t num_class ) 
  {
    this -> num_class_ = num_class;
    this -> Train(inputs, labels);
  }
  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels ) const
  {
    arma::Mat<T> dummy;
    Classify(inputs, labels, dummy);
  }

  /**
   * @param inputs  : X*
   * @param labels  : y*
   * @param scores  : probabilities for the labels
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels,
                  arma::Mat<T>& scores ) const
  {
    scores.resize(num_class_,inputs.n_cols);
    if (ulab_.n_elem != 1 && ulab_.n_elem < 3)
    {
      model_.Classify(inputs, labels, scores);
      scores.resize(num_class_,scores.n_cols);
    }
    else if (ulab_.n_elem == 1)
    {
      scores.row(ulab_[0]).fill(1.);
      labels.resize(inputs.n_cols);
      labels.fill(ulab_[0]);
    }
    else
    {
      ova_.Classify(inputs,labels,scores);
    }
  }

  /**
   * Calculate the Error Rate
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  T ComputeError ( const arma::Mat<T>& points, 
                   const arma::Row<size_t>& responses ) 
  {
    arma::Row<size_t> predictions;
    Classify(points,predictions);
    arma::Row<size_t> temp =  predictions - responses; 
    return (arma::accu(temp != 0))/T(predictions.n_elem);
  }
  /**
   * Calculate the Accuracy
   *
   * @param inputs  : X*
   * @param labels  : y*
   * 
   */
  T ComputeAccuracy ( const arma::Mat<T>& points, 
                      const arma::Row<size_t>& responses ) 
  {
    return (1. - ComputeError(points, responses));
  }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar ( cereal::make_nvp("num_class",num_class_),
         cereal::make_nvp("ulab",ulab_),
         cereal::make_nvp("ova",ova_),
         cereal::make_nvp("model",model_),
         cereal::make_nvp("lambda",lambda_) );

  }

  private:

  size_t num_class_;
  double lambda_;
  arma::Row<size_t> ulab_;

  OnevAll<mlpack::LogisticRegression<arma::Mat<T>>> ova_;
  mlpack::LogisticRegression<arma::Mat<T>> model_;
};

} // namespace algo::classification

#endif
