/**
 * @file manip.h
 * @author Ozgur Taylan Turan
 *
 * Data manipulation related stuff
 *
 */

#ifndef MANIP_H
#define MANIP_H


namespace data { 

//=============================================================================
// SetDiff : Difference between two vectors
//=============================================================================
/**
 * @param check : The vector to be compared
 * @param with  : Comparison is made with this vector
 */
template<class T>
T SetDiff( const T& check, const T& with )
{
  assert ( check.is_sorted() && with.is_sorted() &&
                   "For this method I assumed you sorted your stuff...");

  T result;
  size_t i = 0, j = 0;

  while (i < check.n_elem && j < with.n_elem)
  {
      if (check[i] < with[j])
      {
          result.resize(result.n_elem+1);
          result[result.n_elem-1] = check[i];
          i++;
      }
      else if (check[i] > with[j])
          j++;
      else
      {
          i++;
          j++;
      }
  }

  // Append remaining elements from check
  while (i < check.n_elem)
  {
      result.resize(result.n_elem+1);
      result[result.n_elem-1] = check[i];
      i++;
  }

  return result;
}

//=============================================================================
// Migrate : Exchange N random data points between train and test sets
//=============================================================================
/**
 * @param train_inp : inputs of the training set
 * @param train_lab : labels of the training set
 * @param test_inp  : inputs of the testing set
 * @param test_lab  : labels of the testing set 
 * @param N         : number of points that to be migrated (test->train)
 */
template<typename T, typename U>
void Migrate ( arma::Mat<T>& train_inp,
               arma::Row<U>& train_lab,
               arma::Mat<T>& test_inp,
               arma::Row<U>& test_lab,
               const size_t N )
{
  assert ( ( train_inp.n_cols == train_lab.n_elem &&
                   train_inp.n_rows == test_inp.n_rows &&
                   test_inp.n_cols == test_lab.n_elem &&
                   test_lab.n_elem >= N) && 
                   "Requested element number is bigger than what you have.");

  train_inp.resize(train_inp.n_rows, train_inp.n_cols+N);
  train_lab.resize(train_lab.n_cols+N);
  arma::uvec idx = arma::randperm(test_inp.n_cols, N);
  train_inp.tail_cols(N) = test_inp.cols(idx);
  train_lab.tail_cols(N) = test_lab.cols(idx);
  test_lab.shed_cols(idx);
  test_inp.shed_cols(idx);
}

template<typename T, typename U>
void Migrate ( arma::Mat<T>& train_inp,
               arma::Mat<U>& train_lab,
               arma::Mat<T>& test_inp,
               arma::Mat<U>& test_lab,
               const size_t N )
{
  assert (  ( test_inp.n_cols == test_lab.n_elem &&
                   train_inp.n_rows == test_inp.n_rows &&
                   test_lab.n_rows == train_lab.n_rows &&
                   test_lab.n_elem >= N ) &&
                   "Requested element number is bigger than what you have.");

  train_inp.resize(train_inp.n_rows, train_inp.n_cols+N);
  train_lab.resize(train_lab.n_rows, train_lab.n_cols+N);
  arma::uvec idx = arma::randperm(test_inp.n_cols, N);
  train_inp.tail_cols(N) = test_inp.cols(idx);
  train_lab.tail_cols(N) = test_lab.cols(idx);
  test_lab.shed_cols(idx);
  test_inp.shed_cols(idx);
}
/**
 * @param trainset  : training dataset
 * @param testset   : testing dataset
 * @param N         : number of points that to be migrated (test->train)
 */
template<typename T>
void Migrate ( T& trainset,
               T& testset,
               const size_t N )
{
  Migrate(trainset.inputs_,trainset.labels_,testset.inputs_,testset.labels_,N);

  trainset.size_ = trainset.inputs_.n_cols;
  testset.size_ = testset.inputs_.n_cols;

  trainset.num_class_ = arma::unique(trainset.labels_).eval().n_cols;
  testset.num_class_ = arma::unique(testset.labels_).eval().n_cols;
}

template<typename T=arma::uword>
void Migrate ( arma::Col<T>& trainset,
               arma::Col<T>& testset,
               const size_t N )
{
  assert ( testset.n_elem >= N &&
                   "Requested element number is bigger than what you have.");

  trainset.resize(trainset.n_elem+N);
  arma::uvec idx = arma::randperm(testset.n_elem, N);
  trainset.tail(N) = testset.rows(idx);
  testset.shed_rows(idx);
}

//=============================================================================
// Split : Split datasets for a given number of training points
//=============================================================================
/**
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
  const arma::uvec order =
      arma::shuffle(arma::regspace<arma::uvec>(0, input.n_cols - 1));

  trainData = input.cols(order.rows(0,trainNum-1));
  trainLabel = inputLabel.cols(order.rows(0,trainNum-1));

  testData = input.cols(order.rows(trainNum,input.n_cols-1));
  testLabel = inputLabel.cols(order.rows(trainNum,input.n_cols-1));
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
  //  I am going to solve this problem with a tinier bit of code
  const arma::uvec order =
      arma::shuffle(arma::regspace<arma::uvec>(0, input.n_cols - 1));

  trainData = input.cols(order.rows(0,trainNum-1));
  trainLabel = inputLabel.cols(order.rows(0,trainNum-1));

  testData = input.cols(order.rows(trainNum,input.n_cols-1));
  testLabel = inputLabel.cols(order.rows(trainNum,input.n_cols-1));
}

/**
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
  const arma::uvec order =
      arma::shuffle(arma::regspace<arma::uvec>(0, input.n_cols - 1));

  trainData = input.cols( order.head(trainNum) );
  testData = input.cols( order.tail(input.n_cols-trainNum) );

}

template<typename T>
void Split ( const arma::Row<T>& input,
             arma::Row<T>& trainData,
             arma::Row<T>& testData,
             const size_t trainNum )
{
  const arma::uvec order =
      arma::shuffle(arma::regspace<arma::uvec>(0, input.n_cols - 1));

  trainData = input.cols( order.head(trainNum) );
  testData = input.cols( order.tail(input.n_cols-trainNum) );

}

template<typename T>
void Split ( const arma::Col<T>& input,
             arma::Col<T>& trainData,
             arma::Col<T>& testData,
             const size_t trainNum )
{
  const arma::uvec order =
      arma::shuffle(arma::regspace<arma::uvec>(0, input.n_rows- 1));

  trainData = input.rows( order.head(trainNum) );
  testData = input.rows( order.tail(input.n_rows-trainNum) );
}

/**
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

//=============================================================================
// StratifiedSplit : Split datasets for a given number of training points
// in a stratified manner
//=============================================================================
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
  assert ( ( typeid(T) == typeid(Dataset<arma::Row<size_t>>) ||
                   typeid(T) == typeid(oml::Dataset<size_t>)) && 
      "StratifiedSplit can only be used for classification dataset type...");

  trainset = dataset; testset = dataset;

  StratifiedSplit(dataset.inputs_, dataset.labels_,
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

  assert ( (typeid(T) == typeid(Dataset<arma::Row<size_t>>) ||
                   typeid(T) == typeid(oml::Dataset<size_t>)) &&  
      "StratifiedSplit can only be used for classification dataset type...");

  trainset = dataset; testset = dataset;

  mlpack::data::StratifiedSplit(dataset.inputs_, dataset.labels_,
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
#endif

