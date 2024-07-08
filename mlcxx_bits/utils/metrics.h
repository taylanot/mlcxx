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
    model.Classify(data, preds);

    return arma::accu(labels%arma::log(preds)+(1.-labels)%arma::log(1.-preds))
      /preds.n_cols;
  }

  static const bool NeedsMinimization = true;
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
