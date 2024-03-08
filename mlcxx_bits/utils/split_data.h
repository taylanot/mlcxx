/**
 * @file split_data.h
 * @author Ozgur Taylan Turan 
 *
 * Defines Split(), a utility function to split a dataset into a
 * training set and a test set with given number of training points.
 * This is different than the one in MLPACK where you have to prodive test
 * ratio for the split.
 *
 * You should implement a stratified one if need be!
 */
#ifndef SPLIT_DATA_H
#define SPLIT_DATA_H

//#include <mlpack/prereqs.hpp>

namespace utils {
namespace data {
/**
 * Given an input dataset and labels, split into a training set and test set.
 * Example usage below.  This overload places the split dataset into the four
 * output parameters given (trainData, testData, trainLabel, and testLabel).
 *
 * @param input Input dataset to split.
 * @param label Input labels to split.
 * @param trainData Matrix to store training data into.
 * @param testData Matrix to store test data into.
 * @param trainLabel Vector to store training labels into.
 * @param testLabel Vector to store test labels into.
 * @param trainNum number of training points desired.
 */
template<typename T, typename U>
void Split ( const arma::Mat<T>& input,
             const arma::Row<U>& inputLabel,
             arma::Mat<T>& trainData,
             arma::Mat<T>& testData,
             arma::Row<U>& trainLabel,
             arma::Row<U>& testLabel,
             const size_t trainNum )
{
  const size_t trainSize = trainNum;
  const size_t testSize = input.n_cols - trainSize;

  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);
  trainLabel.set_size(trainSize);
  testLabel.set_size(testSize);

  const arma::Col<size_t> order =
      arma::shuffle(arma::linspace<arma::Col<size_t>>(0, input.n_cols - 1,
                                                      input.n_cols));

  for ( size_t i = 0; i != trainSize; ++i )
  {
    trainData.col(i) = input.col(order[i]);
    trainLabel(i) = inputLabel(order[i]);
  }

  for ( size_t i = 0; i != testSize; ++i )
  {
    testData.col(i) = input.col(order[i + trainSize]);
    testLabel(i) = inputLabel(order[i + trainSize]);
  }
}


template<typename T, typename U>
void Split ( const arma::Mat<T>& input,
             const arma::Mat<U>& inputLabel,
             arma::Mat<T>& trainData,
             arma::Mat<T>& testData,
             arma::Mat<U>& trainLabel,
             arma::Mat<U>& testLabel,
             const size_t trainNum )
{
  const size_t trainSize = trainNum;
  const size_t testSize = input.n_cols - trainSize;

  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);
  trainLabel.set_size(inputLabel.n_rows, trainSize);
  testLabel.set_size(inputLabel.n_rows, testSize);

  const arma::Col<size_t> order =
      arma::shuffle(arma::linspace<arma::Col<size_t>>(0, input.n_cols - 1,
                                                      input.n_cols));

  for ( size_t i = 0; i != trainSize; ++i )
  {
    trainData.col(i) = input.col(order[i]);
    trainLabel.col(i) = inputLabel.col(order[i]);
  }

  for ( size_t i = 0; i != testSize; ++i )
  {
    testData.col(i) = input.col(order[i + trainSize]);
    testLabel.col(i) = inputLabel.col(order[i + trainSize]);
  }
}

/**
 * Given an input dataset, split into a training set and test set.
 * Example usage below. This overload places the split dataset into the two
 * output parameters given (trainData, testData).
 *
 * @param input Input dataset to split.
 * @param trainData Matrix to store training data into.
 * @param testData Matrix to store test data into.
 * @param trainNum number of training points desired.
 */
template<typename T>
void Split ( const arma::Mat<T>& input,
             arma::Mat<T>& trainData,
             arma::Mat<T>& testData,
             const size_t trainNum )
{

  const size_t trainSize = trainNum;
  const size_t testSize = input.n_cols - trainSize;

  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);

  const arma::Col<size_t> order =
      arma::shuffle(arma::linspace<arma::Col<size_t>>(0, input.n_cols -1,
                                                      input.n_cols));

  for ( size_t i = 0; i != trainSize; ++i )
    trainData.col(i) = input.col(order[i]);
  for ( size_t i = 0; i != testSize; ++i )
    testData.col(i) = input.col(order[i + trainSize]);

}

/**
 * Given an input dataset and labels, split into a training set and test set.
 * Example usage below.  This overload returns the split dataset as a std::tuple
 * with four elements: an arma::Mat<T> containing the training data, an
 * arma::Mat<T> containing the test data, an arma::Row<U> containing the
 * training labels, and an arma::Row<U> containing the test labels.
 *
 * @code
 * arma::mat input = loadData();
 * arma::Row<size_t> label = loadLabel();
 * auto splitResult = Split(input, label, 0.2);
 * @endcode
 *
 * @param input Input dataset to split.
 * @param label Input labels to split.
 * @param trainNum number of training points desired.
 * @return std::tuple containing trainData (arma::Mat<T>), testData
 *      (arma::Mat<T>), trainLabel (arma::Row<U>), and testLabel (arma::Row<U>).
 */
template<typename T, typename U>
std::tuple<arma::Mat<T>, arma::Mat<T>, arma::Row<U>, arma::Row<U>>
Split ( const arma::Mat<T>& input,
        const arma::Row<U>& inputLabel,
        const size_t trainNum)
{
  arma::Mat<T> trainData;
  arma::Mat<T> testData;
  arma::Row<U> trainLabel;
  arma::Row<U> testLabel;

  Split(input, inputLabel, trainData, testData, trainLabel, testLabel,
      trainNum);

  return std::make_tuple(std::move(trainData),
                         std::move(testData),
                         std::move(trainLabel),
                         std::move(testLabel));
}

template<typename T, typename U>
std::tuple<arma::Mat<T>, arma::Mat<T>, arma::Mat<U>, arma::Mat<U>>
Split ( const arma::Mat<T>& input,
        const arma::Mat<U>& inputLabel,
        const size_t trainNum )
{
  arma::Mat<T> trainData;
  arma::Mat<T> testData;
  arma::Mat<U> trainLabel;
  arma::Mat<U> testLabel;

  Split(input, inputLabel, trainData, testData, trainLabel, testLabel,
      trainNum);

  return std::make_tuple(std::move(trainData),
                         std::move(testData),
                         std::move(trainLabel),
                         std::move(testLabel));
}
/**
 * Given an input dataset, split into a training set and test set.
 * Example usage below.  This overload returns the split dataset as a std::tuple
 * with two elements: an arma::Mat<T> containing the training data and an
 * arma::Mat<T> containing the test data.
 *
 * @code
 * arma::mat input = loadData();
 * auto splitResult = Split(input, 0.2);
 * @endcode
 *
 * @param input Input dataset to split.
 * @param trainNum number of training points desired.
 * @return std::tuple containing trainData (arma::Mat<T>)
 *      and testData (arma::Mat<T>).
 */
template<typename T>
std::tuple<arma::Mat<T>, arma::Mat<T>>
Split ( const arma::Mat<T>& input,
        const size_t trainNum)
{
  arma::Mat<T> trainData;
  arma::Mat<T> testData;
  Split(input, trainData, testData, trainNum);

  return std::make_tuple(std::move(trainData),
                         std::move(testData));
}

/**
 * Given a dataset, split into a training set and test set.
 *
 * @param dataset to be splitted
 * @param trainset to be splitted
 * @param testset to be splitted
 * @param trainNum number of training points 
 */
template<typename T>
void Split ( const T& dataset,
             T& trainset,
             T& testset,
             const size_t trainNum )
{

  trainset = dataset; testset = dataset;

  Split(dataset.inputs_, dataset.labels_,
        trainset.inputs_, testset.inputs_,
        trainset.labels_, testset.labels_, trainNum);

  trainset.size_ = trainset.inputs_.n_cols;
  testset.size_ = testset.inputs_.n_cols;

  trainset.dimension_ = trainset.inputs_.n_rows;
  testset.dimension_ = testset.inputs_.n_rows;
}

/**
 * Given a dataset, split into a training set and test set.
 *
 * @param dataset to be splitted
 * @param trainset to be filled
 * @param testset to be filled
 * @param testRatio percentage of test set
 */
template<typename T>
void Split ( const T& dataset,
             T& trainset,
             T& testset,
             const double testRatio )
{

  trainset = dataset; testset = dataset;

  mlpack::data::Split(dataset.inputs_, dataset.labels_,
        trainset.inputs_, testset.inputs_,
        trainset.labels_, testset.labels_, testRatio);

  trainset.size_ = trainset.inputs_.n_cols;
  testset.size_ = testset.inputs_.n_cols;

  trainset.dimension_ = trainset.inputs_.n_rows;
  testset.dimension_ = testset.inputs_.n_rows;
}

/**
 * @param input Input dataset to stratify.
 * @param inputLabel Input labels to stratify.
 * @param trainData Matrix to store training data into.
 * @param testData Matrix to store test data into.
 * @param trainLabel Vector to store training labels into.
 * @param testLabel Vector to store test labels into.
 * @param trainNum number of training points of dataset to use for test set 
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *     sample is visited in linear order. (Default true.)
 */
template<typename T, typename LabelsType,
         typename = std::enable_if_t<arma::is_arma_type<LabelsType>::value> >
void StratifiedSplit(const arma::Mat<T>& input,
                     const LabelsType& inputLabel,
                     arma::Mat<T>& trainData,
                     arma::Mat<T>& testData,
                     LabelsType& trainLabel,
                     LabelsType& testLabel,
                     const size_t trainNum,
                     const bool shuffleData = true)
{
  const bool typeCheck = (arma::is_Row<LabelsType>::value)
      || (arma::is_Col<LabelsType>::value);
  if (!typeCheck)
    throw std::runtime_error("data::Split(): when stratified sampling is done, "
        "labels must have type `arma::Row<>`!");
  mlpack::util::CheckSameSizes(input, inputLabel, "data::Split()");

  double testRatio = double(1) - double(trainNum)/double(inputLabel.n_elem);
  size_t trainIdx = 0;
  size_t testIdx = 0;
  size_t trainSize = 0;
  size_t testSize = 0;
  arma::uvec labelCounts;
  arma::uvec testLabelCounts;
  typename LabelsType::elem_type maxLabel = inputLabel.max();

  labelCounts.zeros(maxLabel+1);
  testLabelCounts.zeros(maxLabel+1);

  for (typename LabelsType::elem_type label : inputLabel)
    ++labelCounts[label];

  for (arma::uword labelCount : labelCounts)
  {
    testSize += floor(labelCount * testRatio+1e-6);
    trainSize += labelCount - floor(labelCount * testRatio+1e-6);
  }

  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);
  trainLabel.set_size(inputLabel.n_rows, trainSize);
  testLabel.set_size(inputLabel.n_rows, testSize);

  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(
        arma::linspace<arma::uvec>(0, input.n_cols - 1, input.n_cols));

    for (arma::uword i : order)
    {
      typename LabelsType::elem_type label = inputLabel[i];
      if (testLabelCounts[label] < floor(labelCounts[label] * testRatio+1e-6))
      {
        testLabelCounts[label] += 1;
        testData.col(testIdx) = input.col(i);
        testLabel[testIdx] = inputLabel[i];
        testIdx += 1;
      }
      else
      {
        trainData.col(trainIdx) = input.col(i);
        trainLabel[trainIdx] = inputLabel[i];
        trainIdx += 1;
      }
    }
  }
  else
  {
    for (arma::uword i = 0; i < input.n_cols; i++)
    {
      typename LabelsType::elem_type label = inputLabel[i];
      if (testLabelCounts[label] < floor(labelCounts[label] * testRatio+1e-6))
      {
        testLabelCounts[label] += 1;
        testData.col(testIdx) = input.col(i);
        testLabel[testIdx] = inputLabel[i];
        testIdx += 1;
      }
      else
      {
        trainData.col(trainIdx) = input.col(i);
        trainLabel[trainIdx] = inputLabel[i];
        trainIdx += 1;
      }
    }
  }
}

/**
 * Unfortunately mlpack has an issue if you have a balanced dataset, so got it
 * from there with a minor fix
 *
 * @param input Input dataset to stratify.
 * @param inputLabel Input labels to stratify.
 * @param trainData Matrix to store training data into.
 * @param testData Matrix to store test data into.
 * @param trainLabel Vector to store training labels into.
 * @param testLabel Vector to store test labels into.
 * @param testRatio ratio of test set
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *     sample is visited in linear order. (Default true.)
 */
template<typename T, typename LabelsType,
         typename = std::enable_if_t<arma::is_arma_type<LabelsType>::value> >
void StratifiedSplit(const arma::Mat<T>& input,
                     const LabelsType& inputLabel,
                     arma::Mat<T>& trainData,
                     arma::Mat<T>& testData,
                     LabelsType& trainLabel,
                     LabelsType& testLabel,
                     const double testRatio,
                     const bool shuffleData = true)
{
  const bool typeCheck = (arma::is_Row<LabelsType>::value)
      || (arma::is_Col<LabelsType>::value);
  if (!typeCheck)
    throw std::runtime_error("data::Split(): when stratified sampling is done, "
        "labels must have type `arma::Row<>`!");
  mlpack::util::CheckSameSizes(input, inputLabel, "data::Split()");

  size_t trainIdx = 0;
  size_t testIdx = 0;
  size_t trainSize = 0;
  size_t testSize = 0;
  arma::uvec labelCounts;
  arma::uvec testLabelCounts;
  typename LabelsType::elem_type maxLabel = inputLabel.max();

  labelCounts.zeros(maxLabel+1);
  testLabelCounts.zeros(maxLabel+1);

  for (typename LabelsType::elem_type label : inputLabel)
    ++labelCounts[label];

  for (arma::uword labelCount : labelCounts)
  {
    testSize += floor(labelCount * testRatio+1e-6);
    trainSize += labelCount - floor(labelCount * testRatio+1e-6);
  }

  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);
  trainLabel.set_size(inputLabel.n_rows, trainSize);
  testLabel.set_size(inputLabel.n_rows, testSize);

  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(
        arma::linspace<arma::uvec>(0, input.n_cols - 1, input.n_cols));

    for (arma::uword i : order)
    {
      typename LabelsType::elem_type label = inputLabel[i];
      if (testLabelCounts[label] < floor(labelCounts[label] * testRatio+1e-6))
      {
        testLabelCounts[label] += 1;
        testData.col(testIdx) = input.col(i);
        testLabel[testIdx] = inputLabel[i];
        testIdx += 1;
      }
      else
      {
        trainData.col(trainIdx) = input.col(i);
        trainLabel[trainIdx] = inputLabel[i];
        trainIdx += 1;
      }
    }
  }
  else
  {
    for (arma::uword i = 0; i < input.n_cols; i++)
    {
      typename LabelsType::elem_type label = inputLabel[i];
      if (testLabelCounts[label] < floor(labelCounts[label] * testRatio+1e-6))
      {
        testLabelCounts[label] += 1;
        testData.col(testIdx) = input.col(i);
        testLabel[testIdx] = inputLabel[i];
        testIdx += 1;
      }
      else
      {
        trainData.col(trainIdx) = input.col(i);
        trainLabel[trainIdx] = inputLabel[i];
        trainIdx += 1;
      }
    }
  }
}

/**
 * Given a dataset, split into a training set and test set with stratification
 *
 * @param dataset to be splitted
 * @param trainset to be splitted
 * @param testset to be splitted
 * @param trainNum number of training points 
 */
template<typename T>
void StratifiedSplit ( const T& dataset,
                        T& trainset,
                        T& testset,
                        const size_t trainNum )
{
  BOOST_ASSERT_MSG(typeid(T) == typeid(classification::Dataset), 
      "StratifiedSplit can only be used for classification dataset type...");

  trainset = dataset; testset = dataset;

  Split(dataset.inputs_, dataset.labels_,
        trainset.inputs_, testset.inputs_,
        trainset.labels_, testset.labels_, trainNum);

  trainset.size_ = trainset.inputs_.n_cols;
  testset.size_ = testset.inputs_.n_cols;

  trainset.dimension_ = trainset.inputs_.n_rows;
  testset.dimension_ = testset.inputs_.n_rows;
}

/**
 * Given a dataset, split into a training set and test set with stratification
 *
 * @param dataset to be splitted
 * @param trainset to be filled
 * @param testset to be filled
 * @param testRatio percentage of test set
 */
template<typename T>
void StratifiedSplit ( const T& dataset,
             T& trainset,
             T& testset,
             const double testRatio )
{

  BOOST_ASSERT_MSG(typeid(T) == typeid(classification::Dataset), 
      "StratifiedSplit can only be used for classification dataset type...");

  trainset = dataset; testset = dataset;

  mlpack::data::Split(dataset.inputs_, dataset.labels_,
        trainset.inputs_, testset.inputs_,
        trainset.labels_, testset.labels_, testRatio);

  trainset.size_ = trainset.inputs_.n_cols;
  testset.size_ = testset.inputs_.n_cols;

  trainset.dimension_ = trainset.inputs_.n_rows;
  testset.dimension_ = testset.inputs_.n_rows;
}

template<typename T, typename U>
std::tuple<arma::Mat<T>, arma::Mat<T>, arma::Row<U>, arma::Row<U>>
StratifiedSplit ( const arma::Mat<T>& input,
                  const arma::Row<U>& inputLabel,
                  const size_t trainNum)
{
  arma::Mat<T> trainData;
  arma::Mat<T> testData;
  arma::Row<U> trainLabel;
  arma::Row<U> testLabel;

  StratifiedSplit(input, inputLabel, trainData, testData, trainLabel, testLabel,
      trainNum);

  return std::make_tuple(std::move(trainData),
                         std::move(testData),
                         std::move(trainLabel),
                         std::move(testLabel));
}



} // namespace data
} // namespace utils

#endif
