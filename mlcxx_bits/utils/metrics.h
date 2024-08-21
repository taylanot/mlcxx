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
  template<typename MLAlgorithm, typename DataType, typename LabelType, class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    // Get the predicted probabilities
    LabelType preds;
    arma::Mat<O> probs;
    model.Classify(data, preds, probs);
    probs.clamp(std::numeric_limits<O>::epsilon(),1.-std::numeric_limits<O>::epsilon());
    LabelType labs = arma::clamp(labels,1.e-16,1.-1.e-16);

    return -arma::accu( labs % arma::log(arma::max(probs,0))
                      +(1.-labs)%arma::log(1.-arma::max(probs,0))
                      ) /preds.n_cols;
  }

  static const bool NeedsMinimization = true;
};

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
  template<typename MLAlgorithm, typename DataType, typename LabelType, class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const LabelType& labels )
  {
    // Get the predicted probabilities
    LabelType preds;
    model.Classify(data, preds);
    return preds(0,0);
  }
};

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
  template<typename MLAlgorithm, typename DataType, typename LabelType, class O=DTYPE>
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
  template<typename MLAlgorithm, typename DataType, typename LabelType,class O=DTYPE>
  static O Evaluate( MLAlgorithm& model,
                     const DataType& data,
                     const arma::Row<size_t>& labels )
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
