/**
 * @file xval.h
 * @author Ozgur Taylan Turan
 *
 * I will try to expend the functionalities of the cross-validation of mlpack.
 * Doin this to get the error distribution out of it!
 */

namespace xval2 {

template<typename MLAlgorithm,
         typename Metric,
         typename MatType = arma::Mat<DTYPE>,
         typename PredictionsType =
             typename mlpack::MetaInfoExtractor<MLAlgorithm,MatType>::PredictionsType,
         typename WeightsType =
             typename mlpack::MetaInfoExtractor<MLAlgorithm, MatType,
                 PredictionsType>::WeightsType,
         typename T = DTYPE>
class KFoldCV : public mlpack::KFoldCV<MLAlgorithm,Metric,MatType,PredictionsType,WeightsType>
{

public:

  template <typename... Args>
  KFoldCV (Args... args) : mlpack::KFoldCV<MLAlgorithm,Metric,MatType,PredictionsType,WeightsType>(args...)
  {
    auto argsTuple = std::make_tuple(args...);
    k_ = std::get<0>(argsTuple);
    results_.set_size(k_);

    auto xsOrig = std::get<1>(argsTuple);
    auto ysOrig = std::get<2>(argsTuple);
    if (std::get<3>(argsTuple))
      mlpack::ShuffleData(xsOrig, ysOrig, xsOrig, ysOrig);

    InitKFoldCVMat(xsOrig,x_);
    InitKFoldCVMat(ysOrig,y_);
  }


public:
  template<typename... MLAlgorithmArgs>
  arma::Row<T> Validate(const MLAlgorithmArgs&... args) 
  {  
    size_t numInvalidScores = 0;
    for (size_t i = 0; i < k_; ++i)
    {
      MLAlgorithm&& model  = base.Train(GetTrainingSubset(x_, i),
          GetTrainingSubset(y_, i), args...);
      PRINT_VAR(GetValidationSubset(y_, i));
      results_(i) = Metric::Evaluate(model, GetValidationSubset(x_, i),
          GetValidationSubset(y_, i));
      if (std::isnan(results_(i)) || std::isinf(results_(i)))
        ++numInvalidScores;
      if (i == k_ - 1)
        modelPtr.reset(new MLAlgorithm(std::move(model)));
    }

    if (numInvalidScores == k_)
      LOG("KFoldCV::TrainAndEvaluate(): all folds returned invalid scores!\
            Returning 0.0 as overall score.");

    PRINT(arma::mean(results_));
    return results_;

  }

private:
  template<typename O>
  void InitKFoldCVMat ( const O& source,
                              O& destination )
  {
    binSize_ = source.n_cols / k_;
    lastBinSize_ = source.n_cols - ((k_ - 1) * binSize_);

    destination = (k_ == 2) ? source : join_rows(source,
        source.cols(0, source.n_cols - lastBinSize_ - 1));
  }

  size_t ValidationSubsetFirstCol( const size_t i )
  {
    return (i == 0) ? binSize_ * (k_ - 1) : binSize_ * (i - 1);
  }

  template<typename O>
  arma::Mat<O> GetTrainingSubset ( arma::Mat<O>& m, const size_t i )
  {
    const size_t subsetSize = (i != 0) ? lastBinSize_ + (k_ - 2) * binSize_ :
        (k_ - 1) * binSize_;

    return arma::Mat<O>(m.colptr(binSize_ * i), m.n_rows, subsetSize,
        false, true);
  }

  template<typename O>
  arma::Row<O> GetTrainingSubset ( arma::Row<O>& r, const size_t i )
  {
    const size_t subsetSize = (i != 0) ? lastBinSize_ + (k_ - 2) * binSize_ :
        (k_ - 1) * binSize_;

    return arma::Row<O>(r.colptr(binSize * i), subsetSize, false, true);
  }

  template<typename O>
  arma::Mat<O> GetValidationSubset ( arma::Mat<O>& m, const size_t i )
  {
    const size_t subsetSize = (i == 0) ? lastBinSize_ : binSize_;
    return arma::Mat<O>(m.colptr(ValidationSubsetFirstCol(i)), m.n_rows,
        subsetSize, false, true);
  }


  template<typename O>
  arma::Row<O> GetValidationSubset ( arma::Row<O>& r, const size_t i )
  {
    const size_t subsetSize = (i == 0) ? lastBinSize_ : binSize_;
    return arma::Row<O>(r.colptr(ValidationSubsetFirstCol(i)),
        subsetSize, false, true);
  }


  arma::Row<T> results_;
  size_t k_;
  size_t binSize_;
  size_t lastBinSize_;
  //! An auxiliary object.
  using Base = mlpack::CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;
  Base base;

  //! The extended (by repeating the first k - 2 bins) data points.
  MatType x_;
  //! The extended (by repeating the first k - 2 bins) predictions.
  PredictionsType y_;

  //! The original size of the dataset.
  size_t lastBinSize;

  //! The size of each bin in terms of data points.
  size_t binSize;

  //! A pointer to a model from the last run of k-fold cross-validation.
  std::unique_ptr<MLAlgorithm> modelPtr;

};

}
