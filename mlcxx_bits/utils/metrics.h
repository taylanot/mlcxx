/**
 * @file metrics.h
 * @author Ozgur Taylan Turan
 *
 * Some metrics that I miss in mlpack
 *
 * TODO: 
 *
 *
 */

#ifndef METRICS_H
#define METRICS_H

namespace utils {
//=============================================================================
// LogLoss : Only binary classification task, see cross entropy for multi-class
//=============================================================================
class LogLoss 
{
public:
  /**
   * Run classification and calculate cross-entropy loss.
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm,
           typename DataType,
           typename LabelType,
           class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    mlpack::util::CheckSameSizes(data,(size_t) labels.n_cols,
        "MSE::Evaluate()",
        "responses");
    LabelType preds;
    arma::Mat<O> probs;
    model.Classify(data, preds, probs);
    probs.clamp(std::numeric_limits<O>::epsilon(),
                1.-std::numeric_limits<O>::epsilon());
    BOOST_ASSERT_MSG((size_t) probs.n_rows == 2, 
                      "LogLoss : Not binary classification." );
    /* LabelType labs = arma::clamp(labels,1.e-16,1.-1.e-16); */
    return -arma::accu( labels % arma::log(arma::max(probs,0))
                      +(1.-labels)%arma::log(1.-arma::max(probs,0))
                      ) /preds.n_cols;
  }

  static const bool NeedsMinimization = true;
};

//=============================================================================
// CrossEntropy : Multi-class LogLoss
//=============================================================================
class CrossEntropy
{
public:
  /**
   * Run classification and calculate cross-entropy loss.
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm,
           typename DataType,
           typename LabelType,
           class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    mlpack::util::CheckSameSizes(data,(size_t) labels.n_cols,
        "MSE::Evaluate()",
        "responses");
    // Get the encoded version of the labels
    arma::Mat<DTYPE> encoded;
    mlpack::data::OneHotEncoding(labels,encoded);
    LabelType preds;
    arma::Mat<O> probs;
    model.Classify(data, preds, probs);
    // For numerical stability with the probability
    probs.clamp(std::numeric_limits<O>::epsilon(),
                1.-std::numeric_limits<O>::epsilon());
    return -arma::accu(encoded%arma::log(probs))/double(labels.n_cols); 
  }

  static const bool NeedsMinimization = true;
};

//=============================================================================
// AUC : Area Under Curve Score
//=============================================================================
class AUC
{
public:
  /**
   * Run classification and calculate area under the curve score.
   * Works for multi-class cases as well.
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm,
           typename DataType,
           typename LabelType,
           class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    mlpack::util::CheckSameSizes(data,(size_t) labels.n_cols,
        "MSE::Evaluate()",
        "responses");
    arma::Mat<O> probs;
    arma::Row<size_t> preds;
    model.Classify(data, preds, probs);
    arma::Row<size_t> unq = arma::unique(labels);
    mlpack::ROCAUCScore<1> auc;
    arma::rowvec aucscores(unq.n_elem);
    for (size_t i=0;i<unq.n_elem;i++)
    {
      auto binlabels = arma::conv_to<arma::Row<size_t>>::from(labels==unq(i));
      aucscores(i) = auc.Evaluate(binlabels,probs.row(unq(i)));
    }
    return arma::mean(aucscores); 
  }

  static const bool NeedsMinimization = false;
};

//=============================================================================
// BrierLoss : Mean Squared Error like measure for classification
//=============================================================================
class BrierLoss
{
public:
  /**
   * Run classification and calculate Brier Loss
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm,
           typename DataType,
           typename LabelType,
           class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    mlpack::util::CheckSameSizes(data,(size_t) labels.n_cols,
        "MSE::Evaluate()",
        "responses");
    arma::Mat<DTYPE> encoded;
    mlpack::data::OneHotEncoding(labels,encoded);
    LabelType preds;
    arma::Mat<O> probs;
    model.Classify(data, preds, probs);
    PRINT(arma::accu(arma::pow(encoded-probs,2))/(labels.n_elem*probs.n_rows));
    return arma::accu(arma::pow(encoded-probs,2))/(labels.n_elem*probs.n_rows);
  }

  static const bool NeedsMinimization = true;
};

//=============================================================================
// DummyClass : This just returns the first prediction.
//=============================================================================
class DummyClass
{
public:
  /**
   * Run classification and calculate Nothing
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm,
           typename DataType,
           typename LabelType,
           class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    LabelType preds;
    model.Classify(data, preds);
    return preds(0,0);
  }
};

//=============================================================================
// DummyClass : This just returns the first prediction.
//=============================================================================
class DummyReg
{
public:
  /**
   * Run predict and calculate Nothing
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm,
           typename DataType,
           typename LabelType,
           class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    // Get the predicted probabilities
    LabelType preds;
    model.Predict(data, preds);
    // return only the first element assuming you have only one variable
    // you would like to observe
    return preds(0,0);
  }
};

//=============================================================================
// ErrorRate : Just 1-Accuracy
//=============================================================================
class ErrorRate
{
 public:
  /**
   * Run classification and calculate error rate.
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm, typename DataType,
           typename LabelType, class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    mlpack::Accuracy acc;
    return 1. - acc.Evaluate(model,data,labels);
  }
  static const bool NeedsMinimization = true;
};

//=============================================================================
// MSEClass : This is Mean squared error for the classification problems, since
//            your problem is know dealing with Row<size_t> instead of Row<T>.
//=============================================================================
class MSEClass 
{
 public:
  /**
   * Run classification and calculate Mean Squared Error.
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm,
           typename DataType,
           typename LabelType,
           class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    mlpack::util::CheckSameSizes(data,(size_t) labels.n_cols,
        "MSE::Evaluate()",
        "responses");

    LabelType predictedResponses;
    model.Classify(data, predictedResponses);
    double sum = arma::accu(arma::square(labels - predictedResponses));
    return sum / labels.n_elem;
  }

  static const bool NeedsMinimization = true;
};

} // namespace utils

#endif 
