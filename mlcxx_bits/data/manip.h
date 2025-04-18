/**
 * @file manip.h
 * @author Ozgur Taylan Turan
 *
 * Data manipulation related stuff
 *
 * TODO: 
 *
 *
 */

#ifndef MANIP_H
#define MANIP_H


namespace data { 

//=============================================================================
// SetDiff : Difference between two vectors
//=============================================================================
template<class T>
T SetDiff( const T& check, const T& with )
{
  BOOST_ASSERT_MSG( check.is_sorted() && with.is_sorted(),
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
          // a[i] == b[j], skip this element
          i++;
          j++;
      }
  }

  // Append remaining elements from a
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
template<typename T, typename U>
void Migrate ( arma::Mat<T>& train_inp,
               arma::Row<U>& train_lab,
               arma::Mat<T>& test_inp,
               arma::Row<U>& test_lab,
               const size_t N )
             
{

  BOOST_ASSERT_MSG(train_inp.n_cols == train_lab.n_elem &&
                   train_inp.n_rows == test_inp.n_rows &&
                   test_inp.n_cols == test_lab.n_elem &&
                   test_lab.n_elem >= N, 
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
  BOOST_ASSERT_MSG(test_inp.n_cols == test_lab.n_elem &&
                   train_inp.n_rows == test_inp.n_rows &&
                   test_lab.n_rows == train_lab.n_rows &&
                   test_lab.n_elem >= N,
                   "Requested element number is bigger than what you have.");

  train_inp.resize(train_inp.n_rows, train_inp.n_cols+N);
  train_lab.resize(train_lab.n_rows, train_lab.n_cols+N);
  arma::uvec idx = arma::randperm(test_inp.n_cols, N);
  train_inp.tail_cols(N) = test_inp.cols(idx);
  train_lab.tail_cols(N) = test_lab.cols(idx);
  test_lab.shed_cols(idx);
  test_inp.shed_cols(idx);
}

template<typename T>
void Migrate ( const T& dataset,
               T& trainset,
               T& testset,
               const size_t N )
             
{
  Migrate(trainset.inputs_,trainset.labels_,testset.inputs_,testset.labels_,N);

  trainset.size_ = trainset.inputs_.n_cols;
  testset.size_ = testset.inputs_.n_cols;

  trainset.num_class_ = arma::unique(trainset.labels_).eval().n_cols;
  testset.num_class_ = arma::unique(testset.labels_).eval().n_cols;
}
//=============================================================================
// select_header : Select the relavent parts of the database by using 
//                  arma::field
//=============================================================================
arma::uvec select_header(const arma::field<std::string>& header,
                         const std::string& which)
{

  size_t count=0;
  for (size_t j = 0; j < header.n_elem; j++)
  {
    bool check = header(j).find(which) != std::string::npos;
    if (check)
      count++;
  }

  arma::uvec ids(count);
  count = 0;
  for (size_t j = 0; j < header.n_elem; j++)
  {
    bool check = header(j).find(which) != std::string::npos;
    if (check)
      ids(count++) = j;
  }
  return ids;
}

//=============================================================================
// select_header : Selecting multiple relavent parts and finding the 
//                  intersection between those by using arma::field
//=============================================================================
arma::uvec select_headers(const arma::field<std::string>& header,
                          const std::vector<std::string>& whichs)
{
  BOOST_ASSERT_MSG ( whichs.size() > 1,
      "The size of the search vector should be more than 1!");

  arma::field<arma::uvec> ids(1,whichs.size());

  size_t counter = 0;
  for (std::string which: whichs)
    ids(0,counter++) = select_header(header,which);

  arma::uvec id;
  for (size_t i=0; i<whichs.size()-1;i++)
  {
    if (i ==0)
      id = arma::intersect(ids(i),ids(i+1));
    else
      id = arma::intersect(id,ids(i+1));

  }


  return id;
}


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
  /* const size_t trainSize = trainNum; */
  /* const size_t testSize = input.n_cols - trainSize; */

  /* trainData.set_size(input.n_rows, trainSize); */
  /* testData.set_size(input.n_rows, testSize); */
  /* trainLabel.set_size(trainSize); */
  /* testLabel.set_size(testSize); */

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
  /* const size_t trainSize = trainNum; */
  /* const size_t testSize = input.n_cols - trainSize; */

  /* trainData.set_size(input.n_rows, trainSize); */
  /* testData.set_size(input.n_rows, testSize); */
  /* trainLabel.set_size(inputLabel.n_rows, trainSize); */
  /* testLabel.set_size(inputLabel.n_rows, testSize); */

  /* const arma::Col<size_t> order = */
  /*     arma::shuffle(arma::linspace<arma::Col<size_t>>(0, input.n_cols - 1, */
  /*                                                     input.n_cols)); */

  /* for ( size_t i = 0; i != trainSize; ++i ) */
  /* { */
  /*   trainData.col(i) = input.col(order[i]); */
  /*   trainLabel.col(i) = inputLabel.col(order[i]); */
  /* } */

  /* for ( size_t i = 0; i != testSize; ++i ) */
  /* { */
  /*   testData.col(i) = input.col(order[i + trainSize]); */
  /*   testLabel.col(i) = inputLabel.col(order[i + trainSize]); */
  /* } */
  const arma::uvec order =
      arma::shuffle(arma::regspace<arma::uvec>(0, input.n_cols - 1));

  trainData = input.cols(order.rows(0,trainNum-1));
  trainLabel = inputLabel.cols(order.rows(0,trainNum-1));

  testData = input.cols(order.rows(trainNum,input.n_cols-1));
  testLabel = inputLabel.cols(order.rows(trainNum,input.n_cols-1));
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
  const arma::uvec order =
      arma::shuffle(arma::regspace<arma::uvec>(0, input.n_cols - 1));

  trainData = input.cols(order.rows(0,trainNum-1));

  testData = input.cols(order.rows(trainNum,input.n_cols-1));

  /* const size_t trainSize = trainNum; */
  /* const size_t testSize = input.n_cols - trainSize; */

  /* trainData.set_size(input.n_rows, trainSize); */
  /* testData.set_size(input.n_rows, testSize); */

  /* const arma::Col<size_t> order = */
  /*     arma::shuffle(arma::linspace<arma::Col<size_t>>(0, input.n_cols -1, */
  /*                                                     input.n_cols)); */

  /* for ( size_t i = 0; i != trainSize; ++i ) */
  /*   trainData.col(i) = input.col(order[i]); */
  /* for ( size_t i = 0; i != testSize; ++i ) */
  /*   testData.col(i) = input.col(order[i + trainSize]); */

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
  BOOST_ASSERT_MSG(typeid(T) == typeid(classification::Dataset<>) ||
                   typeid(T) == typeid(oml::Dataset<size_t>) , 
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

  BOOST_ASSERT_MSG(typeid(T) == typeid(classification::Dataset<>) ||
                   typeid(T) == typeid(oml::Dataset<size_t>) , 
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

class N_Split
{
public:
  template<class... Args>
  auto operator()(Args... args)
  {
    return data::Split(args...);
  }
};

class P_Split
{
public:
  template<class... Args>
  auto operator()(Args... args)
  {
    return mlpack::data::Split(args...);
  }
};

class P_StratSplit
{
public:
  template<class... Args>
  auto operator()(Args... args)
  {
    return mlpack::data::StratifiedSplit(args...);
  }
};


class N_StratSplit
{
public:
  template<class... Args>
  auto operator()(Args... args)
  {
    return data::StratifiedSplit(args...);
  }
};

/* //============================================================================= */
/* // Add : Add N random data points between train and test sets */
/* //============================================================================= */
/* template<typename T, typename U> */
/* void Add  ( arma::Mat<T>& train_inp, */
/*             arma::Row<U>& train_lab, */
/*             const arma::Mat<U>& test_inp, */
/*             const arma::Row<U>& test_lab, */
/*             const size_t N ) */
             
/* { */

/*   BOOST_ASSERT_MSG(train_inp.n_cols == train_lab.n_elem && */
/*                    train_inp.n_rows == test_inp.n_rows && */
/*                    test_inp.n_cols == test_lab.n_elem && */
/*                    test_lab.n_elem >= N, */ 
/*                    "Requested element number is bigger than what you have."); */

/*   train_inp.resize(train_inp.n_rows, train_inp.n_cols+N); */
/*   train_lab.resize(train_lab.n_cols+N); */
/*   arma::uvec idx = arma::randperm(test_inp.n_elem, N); */
/*   train_inp.tail_cols(N) = test_inp.cols(idx); */
/*   train_lab.tail_cols(N) = test_lab.cols(idx); */
/* } */

/* template<typename T, typename U> */
/* void Add ( arma::Mat<T>& train_inp, */
/*            arma::Mat<U>& train_lab, */
/*            arma::Mat<T>& test_inp, */
/*            arma::Mat<U>& test_lab, */
/*            const size_t N ) */
             
/* { */
/*   BOOST_ASSERT_MSG(test_inp.n_cols == test_lab.n_elem && */
/*                    train_inp.n_rows == test_inp.n_rows && */
/*                    test_lab.n_rows == train_lab.n_rows && */
/*                    test_lab.n_elem >= N, */
/*                    "Requested element number is bigger than what you have."); */

/*   train_inp.resize(train_inp.n_rows, train_inp.n_cols+N); */
/*   train_lab.resize(train_lab.n_rows, train_lab.n_cols+N); */
/*   arma::uvec idx = arma::randperm(test_inp.n_elem, N); */
/*   train_inp.tail_cols(N) = test_inp.cols(idx); */
/*   train_lab.tail_cols(N) = test_lab.cols(idx); */
/* } */

/**
 * Given a dataset, select N of them randomly without replacement and seperate 
 * the rest
 *
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 */

struct RandomSelect
{
  template<class DATASET,class T>
  std::pair<DATASET,DATASET> operator() ( const DATASET& dataset, const T size )
  {
    DATASET trainset,testset;
    Split(dataset,trainset,testset,size);
    return std::pair<DATASET&,DATASET&>(trainset,testset);
  }

  template<class DATASET,class T>
  void operator()
  ( const DATASET& dataset, const arma::Row<T> sizes, const size_t repeat,
    std::unordered_map<size_t,std::pair<DATASET,DATASET>>& collect,
    size_t& counter )
	{
    for (size_t j = 0; j < repeat; ++j)
      for (size_t i = 0; i < sizes.n_elem; ++i)
        collect[counter++] = (*this)(dataset, sizes[i]); 
	}

};

/**
 * Given a dataset, select N of them randomly with replacement and seperate the 
 * rest.
 *
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 * @param collect : collection of your sets
 * @param counter : for labeling your sets
 */
struct Bootstrap
{
  template<class DATASET,class T>
  std::pair<DATASET,DATASET> operator()(const DATASET& dataset, const T size)
  {
    DATASET trainset,testset;
    auto all = arma::regspace<arma::uvec>(0,1,dataset.labels_.n_cols-1);
    auto sel = arma::randi<arma::uvec>(size,
                                arma::distr_param(0,dataset.labels_.n_cols-1));
    sel = arma::sort(sel);
    trainset.inputs_ = dataset.inputs_.cols(sel); 
    trainset.labels_ = dataset.labels_.cols(sel); 
    auto rest = data::SetDiff(all,sel);
    testset.inputs_ = dataset.inputs_.cols(rest); 
    testset.labels_ = dataset.labels_.cols(rest); 

    return std::pair<DATASET,DATASET>(trainset,testset);
  }

  template<class DATASET,class T>
  void operator()
  ( const DATASET& dataset, const arma::Row<T> sizes, const size_t repeat,
    std::unordered_map<size_t,std::pair<DATASET,DATASET>>& collect,
    size_t& counter )
	{
    for (size_t j = 0; j < repeat; ++j)
      for (size_t i = 0; i < sizes.n_elem; ++i)
        collect[counter++] = (*this)(dataset, sizes[i]); 
	}
};

/**
 * Given a dataset, select N of them randomly and keep on taking form the rest
 *
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 */
struct Additive
{
  
  template<class DATASET,class T>
  void operator()
  ( const DATASET& dataset, const arma::Row<T> sizes, const size_t repeat,
    std::unordered_map<size_t,std::pair<DATASET,DATASET>>& collect,
    size_t& counter )
  {
    for (size_t j = 0; j < repeat; ++j)
    {
      DATASET trainset,testset;
      for (size_t i=0; i < sizes.n_elem; i++)
      {
        if (i == 0)
          Split(dataset,trainset,testset,sizes[i]);
        else
          Migrate(dataset,trainset,testset,sizes[i]-sizes[i-1]);
        collect[counter++] = std::pair<DATASET,DATASET>(trainset,testset);
      }
    }
  }
};

} // namespace data
#endif

