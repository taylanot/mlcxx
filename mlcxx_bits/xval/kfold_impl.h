/*
 * @file kfold_impl.cpp
 * @author Ozgur Taylan TUran
 *
 * Modifying some parts for my own use.
 *
 * License: The original portions derived
 * from mlpack remain subject to the 3-clause BSD license.
 */

#ifndef KFOLD_IMPL_H
#define KFOLD_IMPL_H

namespace xval {

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType,T>::KFoldCV ( const size_t k,
                                  const MatType& xs,
                                  const PredictionsType& ys,
                                  const bool shuffle ) :
    KFoldCV(Base(), k, xs, ys, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType,T>::KFoldCV ( const size_t k,
                                  const MatType& xs,
                                  const PredictionsType& ys,
                                  const size_t numClasses,
                                  const bool shuffle ) :
    KFoldCV(Base(numClasses), k, xs, ys, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType,T>::KFoldCV ( const size_t k,
                                  const MatType& xs,
                                  const mlpack::data::DatasetInfo& datasetInfo,
                                  const PredictionsType& ys,
                                  const size_t numClasses,
                                  const bool shuffle ) :
    KFoldCV(Base(datasetInfo, numClasses), k, xs, ys, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType,T>::KFoldCV ( const size_t k,
                                  const MatType& xs,
                                  const PredictionsType& ys,
                                  const WeightsType& weights,
                                  const bool shuffle ) :
    KFoldCV(Base(), k, xs, ys, weights, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType,T>::KFoldCV ( const size_t k,
                                  const MatType& xs,
                                  const PredictionsType& ys,
                                  const size_t numClasses,
                                  const WeightsType& weights,
                                  const bool shuffle ) :
    KFoldCV(Base(numClasses), k, xs, ys, weights, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType,T>::KFoldCV ( const size_t k,
                                  const MatType& xs,
                                  const mlpack::data::DatasetInfo& datasetInfo,
                                  const PredictionsType& ys,
                                  const size_t numClasses,
                                  const WeightsType& weights,
                                  const bool shuffle ) :
    KFoldCV(Base(datasetInfo, numClasses), k, xs, ys, weights, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType,T>::KFoldCV ( Base&& base,
                                  const size_t k,
                                  const MatType& xs,
                                  const PredictionsType& ys,
                                  const bool shuffle ) :
    base(std::move(base)),
    k(k)
{
  if (k < 2)
    throw std::invalid_argument("KFoldCV: k should not be less than 2");

  Base::AssertDataConsistency(xs, ys);

  InitKFoldCVMat(xs, this->xs);
  InitKFoldCVMat(ys, this->ys);

  // Do we need to shuffle the dataset?
  if (shuffle)
    Shuffle();
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType,T>::KFoldCV ( Base&& base,
                                  const size_t k,
                                  const MatType& xs,
                                  const PredictionsType& ys,
                                  const WeightsType& weights,
                                  const bool shuffle ) :
    base(std::move(base)),
    k(k)
{
  Base::AssertWeightsConsistency(xs, weights);

  InitKFoldCVMat(xs, this->xs);
  InitKFoldCVMat(ys, this->ys);
  InitKFoldCVMat(weights, this->weights);

  // Do we need to shuffle the dataset?
  if (shuffle)
    Shuffle();
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<typename... MLAlgorithmArgs>
double KFoldCV<MLAlgorithm,
               Metric,
               MatType,
               PredictionsType,
               WeightsType,T>::Evaluate ( const MLAlgorithmArgs&... args )
{
  return arma::mean(TrainAndEvaluate(args...));
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
MLAlgorithm& KFoldCV<MLAlgorithm,
                     Metric,
                     MatType,
                     PredictionsType,
                     WeightsType,T>::Model()
{
  if (modelPtr == nullptr)
    throw std::logic_error(
        "KFoldCV::Model(): attempted to access an uninitialized model");

  return *modelPtr;
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<typename DataType>
void KFoldCV<MLAlgorithm,
             Metric,
             MatType,
             PredictionsType,
             WeightsType,T>::InitKFoldCVMat ( const DataType& source,
                                              DataType& destination )
{
  binSize = source.n_cols / k;
  lastBinSize = source.n_cols - ((k - 1) * binSize);

  destination = (k == 2) ? source : join_rows(source,
      source.cols(0, source.n_cols - lastBinSize - 1));
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<typename... MLAlgorithmArgs, bool Enabled, typename>
arma::Row<T> KFoldCV<MLAlgorithm,
                     Metric,
                     MatType,
                     PredictionsType,
                     WeightsType,T>::
                            TrainAndEvaluate ( const MLAlgorithmArgs&... args )
{
  arma::Row<T> evaluations(k);

  size_t numInvalidScores = 0;
  for (size_t i = 0; i < k; i++)
  {
    auto inp = GetTrainingSubset(xs, i);
    auto lab = GetTrainingSubset(ys, i);
    MLAlgorithm&& model  = base.Train(inp, lab, args...);

    evaluations(i) = Metric::Evaluate(model, GetValidationSubset(xs, i),
        GetValidationSubset(ys, i));
    if (std::isnan(evaluations(i)) || std::isinf(evaluations(i)))
    {
      ++numInvalidScores;
      WARNING("KFoldCV::TrainAndEvaluate(): fold " << i << " returned "
          << "a score of " << evaluations(i) << "; ignoring when computing "
          << "the average score.");
    }
    if (i == k - 1)
      modelPtr.reset(new MLAlgorithm(std::move(model)));
  }

  if (numInvalidScores == k)
    WARNING("KFoldCV::TrainAndEvaluate(): all folds returned invalid "
        << "scores!  Returning 0.0 as overall score.");

  return evaluations;
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename>
arma::Row<T> KFoldCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType,T>::TrainAndEvaluate
                                            ( const MLAlgorithmArgs&... args )
{
  arma::Row<DTYPE> evaluations(k);

  for (size_t i = 0; i < k; ++i)
  {
    MLAlgorithm&& model = (weights.n_elem > 0) ?
        base.Train(GetTrainingSubset(xs, i), GetTrainingSubset(ys, i),
            GetTrainingSubset(weights, i), args...) :
        base.Train(GetTrainingSubset(xs, i), GetTrainingSubset(ys, i),
            args...);
    evaluations(i) = Metric::Evaluate(model, GetValidationSubset(xs, i),
        GetValidationSubset(ys, i));
    if (i == k - 1)
      modelPtr.reset(new MLAlgorithm(std::move(model)));
  }

  return evaluations;
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<bool Enabled, typename>
void KFoldCV<MLAlgorithm,
             Metric,
             MatType,
             PredictionsType,
             WeightsType,T>::Shuffle ( )
{
  MatType xsOrig = xs.cols(0, (k - 1) * binSize + lastBinSize - 1);
  PredictionsType ysOrig = ys.cols(0, (k - 1) * binSize + lastBinSize - 1);

  // Now shuffle the data.
  mlpack::ShuffleData(xsOrig, ysOrig, xsOrig, ysOrig);

  InitKFoldCVMat(xsOrig, xs);
  InitKFoldCVMat(ysOrig, ys);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<bool Enabled, typename, typename>
void KFoldCV<MLAlgorithm,
             Metric,
             MatType,
             PredictionsType,
             WeightsType,T>::Shuffle ( )
{
  MatType xsOrig = xs.cols(0, (k - 1) * binSize + lastBinSize - 1);
  PredictionsType ysOrig = ys.cols(0, (k - 1) * binSize + lastBinSize - 1);
  WeightsType weightsOrig;
  if (weights.n_elem > 0)
    weightsOrig = weights.cols(0, (k - 1) * binSize + lastBinSize - 1);

  // Now shuffle the data.
  if (weights.n_elem > 0)
    mlpack::ShuffleData(xsOrig, ysOrig, weightsOrig, xsOrig, ysOrig, weightsOrig);
  else
    mlpack::ShuffleData(xsOrig, ysOrig, xsOrig, ysOrig);

  InitKFoldCVMat(xsOrig, xs);
  InitKFoldCVMat(ysOrig, ys);
  if (weights.n_elem > 0)
    InitKFoldCVMat(weightsOrig, weights);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
size_t KFoldCV<MLAlgorithm,
               Metric,
               MatType,
               PredictionsType,
               WeightsType,T>::ValidationSubsetFirstCol ( const size_t i )
{
  // Use as close to the beginning of the dataset as we can.
  return (i == 0) ? binSize * (k - 1) : binSize * (i - 1);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<typename ElementType>
arma::Mat<ElementType> KFoldCV<MLAlgorithm,
                               Metric,
                               MatType,
                               PredictionsType,
                               WeightsType,T>::GetTrainingSubset 
                                                  ( arma::Mat<ElementType>& m,
                                                    const size_t i )
{
  // If this is not the first fold, we have to handle it a little bit
  // differently, since the last fold may contain slightly more than 'binSize'
  // points.
  const size_t subsetSize = (i != 0) ? lastBinSize + (k - 2) * binSize :
      (k - 1) * binSize;

  return arma::Mat<ElementType>(m.colptr(binSize * i), m.n_rows, subsetSize,
      false, true);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<typename ElementType>
arma::Row<ElementType> KFoldCV<MLAlgorithm,
                               Metric,
                               MatType,
                               PredictionsType,
                               WeightsType,T>::GetTrainingSubset
                                                    ( arma::Row<ElementType>& r,
                                                      const size_t i )
{
  // If this is not the first fold, we have to handle it a little bit
  // differently, since the last fold may contain slightly more than 'binSize'
  // points.
  const size_t subsetSize = (i != 0) ? lastBinSize + (k - 2) * binSize :
      (k - 1) * binSize;

  return arma::Row<ElementType>(r.colptr(binSize * i), subsetSize, false, true);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<typename ElementType>
arma::Mat<ElementType> KFoldCV<MLAlgorithm,
                               Metric,
                               MatType,
                               PredictionsType,
                               WeightsType,T>::GetValidationSubset
                                                    ( arma::Mat<ElementType>& m,
                                                      const size_t i)
{
  const size_t subsetSize = (i == 0) ? lastBinSize : binSize;
  return arma::Mat<ElementType>(m.colptr(ValidationSubsetFirstCol(i)), m.n_rows,
      subsetSize, false, true);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType,
         typename T>
template<typename ElementType>
arma::Row<ElementType> KFoldCV<MLAlgorithm,
                               Metric,
                               MatType,
                               PredictionsType,
                               WeightsType,T>::GetValidationSubset
                                                    ( arma::Row<ElementType>& r,
                                                      const size_t i )
{
  const size_t subsetSize = (i == 0) ? lastBinSize : binSize;
  return arma::Row<ElementType>(r.colptr(ValidationSubsetFirstCol(i)),
      subsetSize, false, true);
}

} // namespace mlpack

#endif
