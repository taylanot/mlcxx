/*
 * @file xkfold.cpp
 * @author Ozgur Taylan TUran
 *
 * Modifying some parts for my own use.
 *
 * License: The original portions derived
 * from mlpack remain subject to the 3-clause BSD license.
 */

#ifndef XK_FOLD_CV_H
#define XK_FOLD_CV_H


namespace xval {

/**
 * The class XKFoldCV implements k-fold cross-validation for regression and
 * classification algorithms and repeats it X times e.g. 5x2 Cross-validation.
 *
 * @tparam MLAlgorithm A machine learning algorithm.
 * @tparam Metric A metric to assess the quality of a trained model.
 * @tparam MatType The type of data.
 * @tparam PredictionsType The type of predictions (should be passed when the
 *     predictions type is a template parameter in Train methods of
 *     MLAlgorithm).
 * @tparam WeightsType The type of weights (should be passed when weighted
 *     learning is supported, and the weights type is a template parameter in
 *     Train methods of MLAlgorithm).
 */
template<typename MLAlgorithm,
         typename Metric,
         typename MatType = arma::Mat<DTYPE>,
         typename PredictionsType =
             typename mlpack::MetaInfoExtractor
                            <MLAlgorithm,MatType>::PredictionsType,
         typename WeightsType =
             typename mlpack::MetaInfoExtractor<MLAlgorithm, MatType,
                 PredictionsType>::WeightsType,
         typename T=DTYPE>
class XKFoldCV
{
 public:
  /**
   * This constructor can be used for regression algorithms and for binary
   * classification algorithms.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param ys Predictions (labels for classification algorithms and responses
   *     for regression algorithms) for each data point.
   * @param repeat how many times?
   */
  XKFoldCV ( const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const size_t repeat );

  /**
   * This constructor can be used for multiclass classification algorithms.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param repeat how many times?
   */
  XKFoldCV ( const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const size_t numClasses,
             const size_t repeat );

  /**
   * This constructor can be used for multiclass classification algorithms that
   * can take a data::DatasetInfo parameter.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param repeat how many times?
   */
  XKFoldCV ( const size_t k,
             const MatType& xs,
             const mlpack::data::DatasetInfo& datasetInfo,
             const PredictionsType& ys,
             const size_t numClasses,
             const size_t repeat );

  /**
   * This constructor can be used for regression and binary classification
   * algorithms that support weighted learning.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param ys Predictions (labels for classification algorithms and responses
   *     for regression algorithms) for each data point.
   * @param weights Observation weights (for boosting).
   * @param repeat how many times?
   */
  XKFoldCV ( const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const WeightsType& weights,
             const bool repeat );

  /**
   * This constructor can be used for multiclass classification algorithms that
   * support weighted learning.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Observation weights (for boosting).
   * @param repeat how many times?
   */
  XKFoldCV ( const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const size_t numClasses,
             const WeightsType& weights,
             const size_t repeat );

  /**
   * This constructor can be used for multiclass classification algorithms that
   * can take a data::DatasetInfo parameter and support weighted learning.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Observation weights (for boosting).
   * @param repeat how many times? 
   */
  XKFoldCV ( const size_t k,
             const MatType& xs,
             const mlpack::data::DatasetInfo& datasetInfo,
             const PredictionsType& ys,
             const size_t numClasses,
             const WeightsType& weights,
             const size_t repeat );

  /**
   * Run k-fold cross-validation.
   *
   * @param args Arguments for MLAlgorithm (in addition to the passed
   *     ones in the constructor).
   */
  template<typename... MLAlgorithmArgs>
  double Evaluate ( const MLAlgorithmArgs& ...args );

  //! Access and modify a model from the last run of k-fold cross-validation.
  MLAlgorithm& Model();

 private:
  //! A short alias for CVBase.
  using Base = mlpack::CVBase<MLAlgorithm,MatType,PredictionsType,WeightsType>;

  /**
   * Shuffle the data.  This overload is called if weights are not supported by
   * the model type.
   */
  template<bool Enabled = !Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>>
  void Shuffle ( );

  /**
   * Shuffle the data.  This overload is called if weights are supported by the
   * model type.
   */
  template<bool Enabled = Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>,
           typename = void>
  void Shuffle ( );

 public:
  //! An auxiliary object.
  Base base;

  //! The number of bins in the dataset.
  const size_t k;

  //! The number of repitions 
  const size_t rep;


  //! The extended (by repeating the first k - 2 bins) data points.
  MatType xs;
  //! The extended (by repeating the first k - 2 bins) predictions.
  PredictionsType ys;
  //! The extended (by repeating the first k - 2 bins) weights.
  WeightsType weights;

  //! The last bin size
  size_t lastBinSize;

  //! The size of each bin in terms of data points.
  size_t binSize;

  //! A pointer to a model from the last run of k-fold cross-validation.
  std::unique_ptr<MLAlgorithm> modelPtr;

  /**
   * Assert the k parameter and data consistency and initialize fields required
   * for running k-fold cross-validation.
   */
  XKFoldCV ( Base&& base,
            const size_t k,
            const MatType& xs,
            const PredictionsType& ys,
            const size_t repeat );

  /**
   * Assert the k parameter and data consistency and initialize fields required
   * for running k-fold cross-validation in the case of weighted learning.
   */
  XKFoldCV ( Base&& base,
            const size_t k,
            const MatType& xs,
            const PredictionsType& ys,
            const WeightsType& weights,
            const size_t repeat );

  /**
   * Initialize the given destination matrix with the given source joined with
   * its first k - 2 bins.
   */
  template<typename DataType>
  void InitXKFoldCVMat ( const DataType& source, DataType& destination );
public:
  /**
   * Train and run evaluation in the case of non-weighted learning.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = !Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>>
  arma::Row<T> TrainAndEvaluate ( const MLAlgorithmArgs& ...mlAlgorithmArgs );

  /**
   * Train and run evaluation in the case of supporting weighted learning.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>,
           typename = void>
  arma::Row<T> TrainAndEvaluate ( const MLAlgorithmArgs& ...mlAlgorithmArgs );

public:
  /**
   * Calculate the index of the first column of the ith validation subset.
   *
   * We take the ith validation subset after the ith training subset if
   * i < k - 1 and before it otherwise.
   */
  inline size_t ValidationSubsetFirstCol ( const size_t i );

  /**
   * Get the ith training subset from a variable of a matrix type.
   */
  template<typename ElementType>
  inline arma::Mat<ElementType> GetTrainingSubset ( arma::Mat<ElementType>& m,
                                                    const size_t i );

  /**
   * Get the ith training subset from a variable of a row type.
   */
  template<typename ElementType>
  inline arma::Row<ElementType> GetTrainingSubset ( arma::Row<ElementType>& r,
                                                    const size_t i );

  /**
   * Get the ith validation subset from a variable of a matrix type.
   */
  template<typename ElementType>
  inline arma::Mat<ElementType> GetValidationSubset ( arma::Mat<ElementType>& m,
                                                     const size_t i );

  /**
   * Get the ith validation subset from a variable of a row type.
   */
  template<typename ElementType>
  inline arma::Row<ElementType> GetValidationSubset( arma::Row<ElementType>& r,
                                                     const size_t i );
};

} // namespace mlpack

// Include implementation
#include "xkfold_impl.h"

#endif
