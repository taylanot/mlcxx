/**
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 * TODO: 
 *
 * - You should put a warning for not using parallelization for neurla networks.
 *   
 *
 *
 */

#ifndef LCURVE_IMPL_NEW_H
#define LCURVE_IMPL_NEW_H

namespace lcurve {

//=============================================================================
// LCurve
//=============================================================================
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,
         class O>
LCurve<MODEL,DATASET,
       SPLIT,LOSS,CV,OPT,O>::LCurve ( const arma::Row<size_t>& Ns,
                                      const size_t repeat,
                                      const bool parallel, 
                                      const bool prog,
                                      const std::string name ) :

repeat_(repeat), Ns_(Ns), parallel_(parallel),
  prog_(prog),name_(name)
{
  _RegisterSignalHandler( );
  _globalSafeFailFunc = [this]() { this->_CleanUp(); };
  test_errors_.resize(repeat_,Ns_.n_elem).fill(arma::datum::nan);
  jobs_ = arma::regspace<arma::uvec>(0,1,size_t(repeat_*Ns_.n_elem)-1);
}

//=============================================================================
// LCurve
//=============================================================================
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,
         class O>
LCurve<MODEL,DATASET,
       SPLIT,LOSS,CV,OPT,O>::  LCurve ( const arma::Row<size_t>& Ns,
                                        const size_t repeat,
                                        const CVP cvp,
                                        const bool parallel, 
                                        const bool prog,
                                        const std::string name) :

repeat_(repeat), Ns_(Ns), parallel_(parallel),
  prog_(prog),name_(name), cvp_(cvp)
{
  _RegisterSignalHandler( );
  _globalSafeFailFunc = [this]() { this->_CleanUp(); };
  test_errors_.resize(repeat_,Ns_.n_elem).fill(arma::datum::nan);
  jobs_ = arma::regspace<arma::uvec>(0,1,size_t(repeat_*Ns_.n_elem)-1);
}

/* Generate Learning Curves 
 *
 * @param dataset   : whole large dataset inputs
 * @param args      : possible arguments for the model initialization
 *
 */
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,
         class O>
template<class... Ts>
void LCurve<MODEL,DATASET,
            SPLIT,LOSS,CV,OPT,O>::Generate ( const DATASET& dataset,
                                             const Ts&... args )
{
  if (data_.empty())
    this->_SplitData(dataset);
  else
    this->_CheckStatus();

  // With this version you cannot use the split version, fyi....
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(dataset.inputs_.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(dataset.labels_.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );

  ProgressBar pb("LCurve.Bootstrap", Ns_.n_elem*repeat_);

  #pragma omp parallel for if (parallel_)
  for (size_t k=0; k < jobs_.n_elem ; k++)
  {
    if constexpr (sizeof...(args) != 0)
    {
      PRINT("IN")
      if (cvp_.has_value())
      {
        auto hpt = _GetHpt(data_[jobs_[k]].first.inputs_,
                           data_[jobs_[k]].first.labels_);

        /* hpt.Optimize(arma::linspace<arma::Row<O>>(0,1,10)); */
        hpt.Optimize(args...);
        MODEL model = std::move(hpt.BestModel());
        model.Train(data_[jobs_[k]].first.inputs_,
                    data_[jobs_[k]].first.labels_);

        test_errors_(jobs_[k]/Ns_.n_elem, jobs_[k] % Ns_.n_elem) = 
                            loss_.Evaluate(model,data_[jobs_[k]].second.inputs_,
                                                 data_[jobs_[k]].second.labels_);
      }
    }
    else
    {
      MODEL model(data_[jobs_[k]].first.inputs_,
                  data_[jobs_[k]].first.labels_, args...);

      test_errors_(jobs_[k]/Ns_.n_elem, jobs_[k] % Ns_.n_elem) = 
                          loss_.Evaluate(model,data_[jobs_[k]].second.inputs_,
                                               data_[jobs_[k]].second.labels_);
    }
    if (prog_)
      pb.Update();
  }
}

/* Generate Learning Curves 
 *
 * @param dataset   : whole large dataset inputs
 * @param args      : possible arguments for the model initialization
 *
 */
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,class O>
template<class... Ts>
void LCurve<MODEL,DATASET,
            SPLIT,LOSS,CV,OPT,O>::Generate ( const DATASET& trainset,
                                             const DATASET& testset,
                                             const Ts&... args )
{
  if (data_.empty())
    this->_SplitData(trainset);
  else
    this->_CheckStatus();

  // With this version you cannot use the split version, fyi....
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(trainset.inputs_.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(trainset.labels_.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );

  ProgressBar pb("LCurve.Bootstrap", Ns_.n_elem*repeat_);

  #pragma omp parallel for if (parallel_)
  for (size_t k=0; k < jobs_.n_elem ; k++)
  {
    MODEL model(data_[jobs_[k]].first.inputs_,
                data_[jobs_[k]].first.labels_, args...);

    test_errors_(jobs_[k]/Ns_.n_elem, jobs_[k] % Ns_.n_elem) = 
                          loss_.Evaluate(model,testset.inputs_,
                                               testset.labels_);
    if (prog_)
      pb.Update();
  }
}

//=============================================================================
// LCurve::_CheckStatus
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,CV,OPT,O>::_CheckStatus(  )
{
  if (jobs_.empty())
    jobs_.clear();
  jobs_ = arma::find_nan( test_errors_ );
}

//=============================================================================
// LCurve::_SplitData
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,CV,OPT,O>::_SplitData( DATASET dataset )
{
  size_t jobid=0;
  split_(dataset,Ns_,repeat_,data_,jobid);
}

//=============================================================================
// LCurve::_RegisterSignalHandler
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,CV,OPT,O>::_RegisterSignalHandler( )
{
  signal(SIGALRM, LCurve<MODEL,DATASET,SPLIT,LOSS,CV,OPT,O>::_SignalHandler);
}

//=============================================================================
// LCurve::_SignalHandler
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,CV,OPT,O>::_SignalHandler( int sig )
{
  if (_globalSafeFailFunc) _globalSafeFailFunc();  
  LOG("\rTime limit exceeded! Exiting..." << std::flush);
  std::quick_exit(0);
}

//=============================================================================
// LCurve::_CleanUp
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,CV,OPT,O>::_CleanUp ( )
{
  LOG("\rCleanUp is called!"<<std::flush);
  Save(name_+".bin");
}

//=============================================================================
// LCurve::Save
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,class O>
void LCurve<MODEL,DATASET,
            SPLIT,LOSS,CV,OPT,O>::Save ( const std::string& filename )
{
  std::ofstream file(filename, std::ios::binary);
  if (!file) 
    ERR("\rCannot open file for writing: " << filename << std::flush);

  cereal::BinaryOutputArchive archive(file);
  archive(cereal::make_nvp("LCurve", *this));  // Serialize the current object
  LOG("\rLCurve object saved to " << filename << std::flush);
}

//=============================================================================
// LCurve::_GetHPT
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,class O>
template<class T>
mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>> 
LCurve<MODEL,DATASET,
       SPLIT,LOSS,CV,OPT,O>::_GetHpt ( const arma::Mat<DTYPE>& Xtrn,
                                       const T& ytrn )
{
  if constexpr (!mlpack::MetaInfoExtractor<MODEL>::TakesNumClasses)
    return mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
      (cvp_.value(), Xtrn, ytrn);
  else
    return mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
      (cvp_.value(), Xtrn, ytrn,size_t(arma::unique(ytrn).eval().n_elem));
}

} // namespace lcurve
#endif
