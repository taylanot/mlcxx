/**
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
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
         class O>
LCurve<MODEL,DATASET,SPLIT,LOSS,O>::LCurve ( const DATASET& dataset,
                                             const arma::Row<size_t>& Ns,
                                             const size_t repeat,
                                             const bool parallel, 
                                             const bool prog,
                                             const std::filesystem::path path,
                                             const std::string name ) :
repeat_(repeat),Ns_(Ns),parallel_(parallel),prog_(prog),name_(name),path_(path)
{
  _RegisterSignalHandler( );
  _globalSafeFailFunc = [this]() { this->_CleanUp(); };
  test_errors_.resize(repeat_,Ns_.n_elem).fill(arma::datum::nan);
  jobs_ = arma::regspace<arma::uvec>(0,1,size_t(repeat_*Ns_.n_elem)-1);

  // With this version you cannot use the split version, fyi....
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(dataset.inputs_.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(dataset.labels_.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );

  if (typeid(dataset.labels_) == typeid(arma::Row<size_t>) ||
        typeid(dataset.labels_) == typeid(arma::Row<int>) )
        num_class_ = dataset.num_class_;

  this->_SplitData(dataset);
}

template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
LCurve<MODEL,DATASET,SPLIT,LOSS,O>::LCurve ( const DATASET& trainset,
                                             const DATASET& testset,
                                             const arma::Row<size_t>& Ns,
                                             const size_t repeat,
                                             const bool parallel, 
                                             const bool prog,
                                             const std::filesystem::path path,
                                             const std::string name ) :
LCurve(trainset,Ns,repeat,parallel,prog,path,name)
{
  testset_= trainset;
}

//=============================================================================
// LCurve::Generate
//=============================================================================
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
template<class... Ts>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::Generate ( const Ts&... args )
{
  if (arma::find_nan(test_errors_).n_elem != jobs_.n_elem)
    this->_CheckStatus();

  ProgressBar pb("LCurve.Generate", jobs_.n_elem);
  #pragma omp parallel for if (parallel_)
  for (size_t k=0; k < jobs_.n_elem ; k++)
  {
    auto model = _GetModel(data_[jobs_[k]].first.inputs_,
                           data_[jobs_[k]].first.labels_,args...);
    
    if (!testset_.has_value())
      test_errors_(jobs_[k]/Ns_.n_elem, jobs_[k] % Ns_.n_elem) = 
                          loss_.Evaluate(model,data_[jobs_[k]].second.inputs_,
                                               data_[jobs_[k]].second.labels_);
    else
      test_errors_(jobs_[k]/Ns_.n_elem, jobs_[k] % Ns_.n_elem) = 
                          loss_.Evaluate(model,testset_.value().inputs_,
                                               testset_.value().labels_);

    if (prog_)
      pb.Update();
  }
}

template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
template<template<class,class,class,class,class> class CV,
         class OPT,
         class T,
         class... Ts>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::Generate ( const T cvp,
                                                    const Ts&... args )
{
  if (arma::find_nan(test_errors_).eval().n_elem != jobs_.n_elem)
    this->_CheckStatus();

  ProgressBar pb("LCurve.Generate", jobs_.n_elem);

  #pragma omp parallel for if (parallel_)
  for (size_t k=0; k < jobs_.n_elem ; k++)
  {
    auto hpt = _GetHpt<CV,OPT>(data_[jobs_[k]].first.inputs_,
                               data_[jobs_[k]].first.labels_,cvp);

    auto best = hpt.Optimize(args...);

    MODEL model = std::apply([&](auto&&... arg) 
    {
      return _GetModel(data_[jobs_[k]].first.inputs_,
                       data_[jobs_[k]].first.labels_,
                       std::forward<decltype(arg)>(arg)...);
    }, best);
 
    if (!testset_.has_value())
      test_errors_(jobs_[k]/Ns_.n_elem, jobs_[k] % Ns_.n_elem) = 
                          loss_.Evaluate(model,data_[jobs_[k]].second.inputs_,
                                               data_[jobs_[k]].second.labels_);
    else
      test_errors_(jobs_[k]/Ns_.n_elem, jobs_[k] % Ns_.n_elem) = 
                          loss_.Evaluate(model,testset_.value().inputs_,
                                               testset_.value().labels_);

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
         class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_CheckStatus(  )
{
  WARN("Finding the incomplete jobs...");
  if (jobs_.empty())
    jobs_.clear();
  jobs_ = arma::find_nan( test_errors_.t() );
}

//=============================================================================
// LCurve::_SplitData
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SplitData ( const DATASET& dataset )
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
         class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_RegisterSignalHandler( )
{
  // for timer related quiting
  signal(SIGALRM, LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SignalHandler);
  // keyboard interupt
  signal(SIGINT, LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SignalHandler);
  // this handles kill
  signal(SIGTERM, LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SignalHandler);
  // this handles kill
  signal(SIGKILL, LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SignalHandler);

}

//=============================================================================
// LCurve::_SignalHandler
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SignalHandler( int sig )
{
  // If one of the signals is detected by our failsafe function,
  // a gracefull exit is initiated...
  if (_globalSafeFailFunc) _globalSafeFailFunc();  
  LOG("\rStopping program for some reason! Exiting..." << std::flush);
  std::quick_exit(0);
}

//=============================================================================
// LCurve::_CleanUp
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_CleanUp ( )
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
         class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::Save ( const std::string& filename )
{
  std::ofstream file((path_/filename), std::ios::binary);
  if (!file) 
    ERR("\rCannot open file for writing: " << (path_/filename) << std::flush);

  // Serialize the object nicely...
  cereal::BinaryOutputArchive archive(file);
  archive(cereal::make_nvp("LCurve", *this));  // Serialize the current object
                                               //
  LOG("\rLCurve object saved to " << (path_/filename) << std::flush);
}

//=============================================================================
// LCurve::Load
//=============================================================================
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
std::shared_ptr<LCurve<MODEL,DATASET,SPLIT,LOSS,O>> 
LCurve<MODEL,DATASET, SPLIT,LOSS,O>::Load ( const std::string& filename )
{
  std::ifstream file(filename, std::ios::binary);
  if (!file) 
  {
    ERR("\rError: Cannot open file for reading: " << filename);
    return nullptr;
  }

  // Deserialize into a new object
  cereal::BinaryInputArchive archive(file);
  auto lcurve = std::make_shared<LCurve<MODEL,DATASET,SPLIT,LOSS,O>>();
  archive(cereal::make_nvp("LCurve", *lcurve));
                                               //
  // These are for making sure the loaded function is safe to fail too...
  _globalSafeFailFunc = [lcurve]() { lcurve->_CleanUp(); };
  lcurve->_RegisterSignalHandler();

  LOG("\rLCurve loaded from " << filename);
  return lcurve;
}

//=============================================================================
// LCurve::_GetHPT
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
template<template<class,class,class,class,class> class CV,
           class OPT,class Tin,class Tlab,class T>
auto LCurve<MODEL,DATASET, SPLIT,LOSS,O>::_GetHpt ( const Tin& Xtrn,
                                                    const Tlab& ytrn,
                                                    const T cvp )
{
  using TunerType = mlpack::HyperParameterTuner<MODEL, LOSS, CV, OPT, Tin>;

  // If we detact the classification algorithm takes in number of classes,
  // another Tuner is called...
  if constexpr (!mlpack::MetaInfoExtractor<MODEL>::TakesNumClasses)
    return TunerType(cvp, Xtrn, ytrn);
  else
    return TunerType(cvp, Xtrn, ytrn,num_class_.value());

}

//=============================================================================
// LCurve::_GetModel
//=============================================================================     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
template<class Tin, class Tlab, class... Ts>
auto LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_GetModel ( const Tin& Xtrn,
                                                     const Tlab& ytrn,
                                                     const Ts&... args )
{
  // If we detact the classification algorithm takes in number of classes,
  // another model initializer is called...
  if constexpr (!mlpack::MetaInfoExtractor<MODEL>::TakesNumClasses)
    return MODEL(Xtrn,ytrn,args...);
  else
    return MODEL(Xtrn,ytrn,num_class_.value(),args...);
}

} // namespace lcurve


namespace cereal
{

	template <class Archive, class T>
	void save(Archive& ar, const std::optional<T>& opt)
	{
    bool hasVal = opt.has_value();
		ar(CEREAL_NVP(hasVal));
		if (hasVal)
		{
			const T& value = *opt;
			ar(cereal::make_nvp("value", value));
		}
	}

	template <class Archive, class T>
	void load(Archive& ar, std::optional<T>& opt)
	{
  	bool hasVal;
		ar(CEREAL_NVP(hasVal));
		if (hasVal)
		{
			T value;
			ar(CEREAL_NVP(value));
			opt = std::move(value);
		}
		else
			opt = std::nullopt;
		
	}

}
#endif
