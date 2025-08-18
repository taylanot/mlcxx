/**
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 */

#ifndef LCURVE_IMPL_NEW_H
#define LCURVE_IMPL_NEW_H

namespace lcurve {

//-----------------------------------------------------------------------------
// LCurve
//-----------------------------------------------------------------------------
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
                                             const std::string name,
                                             const size_t seed ) :
repeat_(repeat),Ns_(Ns),parallel_(parallel),prog_(prog),name_(name),
  path_(path),seed_(seed),trainset_(dataset)
{
  std::filesystem::create_directories(path_);
  _RegisterSignalHandler( );
  _globalSafeFailFunc = [this]() { this->_CleanUp(); };
  test_errors_.resize(repeat_,Ns_.n_elem).fill(arma::datum::inf);

  // With this version you cannot use the split version, fyi....
  assert ( int(Ns_.max()) < int(dataset.inputs_.n_cols) &&
        "There are not enough data for test set creation!" );
  assert ( int(dataset.labels_.n_rows) == int(1) && 
        "Only 1D outputs are allowed!" );

  if (typeid(dataset.labels_) == typeid(arma::Row<size_t>) ||
        typeid(dataset.labels_) == typeid(arma::Row<int>) )
    num_class_ = dataset.num_class_.value();

  // Create the seeds for the jobs
  mlpack::RandomSeed(seed_);
  seeds_ = arma::randi<arma::irowvec>(repeat_,arma::distr_param(0,1000));
}
///////////////////////////////////////////////////////////////////////////////
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
                                             const std::string name,
                                             const size_t seed ) :
LCurve(trainset,Ns,repeat,parallel,prog,path,name)
{
  testset_= trainset;
}
//-----------------------------------------------------------------------------
// LCurve::Generate
//-----------------------------------------------------------------------------
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
template<class... Ts>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::Generate ( const Ts&... args )
{
  if (this->CheckStatus())
    return;

  using DataHolder = std::vector<std::pair<arma::uvec,arma::uvec>>;
  auto run = [&] ( const size_t id, const DataHolder& data ) 
  {
    #pragma omp parallel for if (parallel_) 
    for (size_t k=0; k < data.size() ; k++)
    {
      if (test_errors_(id, k) == arma::datum::inf)
      { 
        auto model = _GetModel(decltype(trainset_.inputs_)
                                (trainset_.inputs_.cols(data[k].first).eval()),
                               decltype(trainset_.labels_)
                                (trainset_.labels_.cols(data[k].first).eval()),
                               args...);

        if (!testset_.has_value())
          test_errors_(id, k) = loss_.Evaluate(model,
                                decltype(trainset_.inputs_)
                                (trainset_.inputs_.cols(data[k].second).eval()),
                                decltype(trainset_.labels_)
                                (trainset_.labels_.cols(data[k].second).eval()));
        else
          test_errors_(id, k) = loss_.Evaluate(model, testset_.value().inputs_,
                                                      testset_.value().labels_);
      }
      else
        continue;
    }

  };
  arma::wall_clock timer;

  utils::ProgressBar pb(name_, repeat_);
  #pragma omp parallel for if (parallel_)
  for (size_t id=0; id<repeat_;id++)
  {
    if (test_errors_.row(id).has_inf())
    {

      auto data = _SplitData(trainset_,seeds_[id]);
      run( id, data );
    }

    if (prog_)
      pb.Update();
  }
  this->Save(name_);
}
//-----------------------------------------------------------------------------
// LCurve::GenerateHpt
//-----------------------------------------------------------------------------
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
template<template<class,class,class,class,class> class CV,
         class OPT,
         class T,
         class... Ts>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::GenerateHpt ( const T cvp,
                                                       const Ts&... args )
{
  if (this->CheckStatus())
    return;

  using DataHolder = std::vector<std::pair<arma::uvec,arma::uvec>>;
  auto run = [&] ( const size_t id, const DataHolder& data ) 
  {
    #pragma omp parallel for if (parallel_) 
    for (size_t k=0; k < data.size() ; k++)
    {
      if (test_errors_(id, k) == arma::datum::inf)
      { 
        auto hpt = _GetHpt<CV,OPT>( (trainset_.inputs_.cols(data[k].first).eval()),
                                (trainset_.labels_.cols(data[k].first).eval()), cvp);

        auto best = hpt.Optimize(args...);

        MODEL model = std::apply([&](auto&&... arg) 
        {
          return _GetModel(decltype(trainset_.inputs_)
                            (trainset_.inputs_.cols(data[k].first).eval()),
                           decltype(trainset_.labels_)
                            (trainset_.labels_.cols(data[k].first).eval()),
                           std::forward<decltype(arg)>(arg)...);
        }, best);

        if (!testset_.has_value())
          test_errors_(id, k) = loss_.Evaluate(model,
                                decltype(trainset_.inputs_)
                              (trainset_.inputs_.cols(data[k].second).eval()),
                                decltype(trainset_.labels_)
                              (trainset_.labels_.cols(data[k].second).eval()));
        else
          test_errors_(id, k) = loss_.Evaluate(model, testset_.value().inputs_,
                                                      testset_.value().labels_);
      }
      else
        continue;
    }

  };

  utils::ProgressBar pb(name_, repeat_);
  #pragma omp parallel for if (parallel_)
  for (size_t id=0; id<repeat_;id++)
  {
    if (test_errors_.row(id).has_inf())
    {
      auto data = _SplitData(trainset_,seeds_[id]);
      run( id, data );
    }

    if (prog_)
      pb.Update();
  }
  this->Save( name_ );
}
//-----------------------------------------------------------------------------
// LCurve::CheckStatus
//-----------------------------------------------------------------------------     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
bool LCurve<MODEL,DATASET,SPLIT,LOSS,O>::CheckStatus( bool print  )
{
  O inc = 0;
  for (size_t i=0; i<repeat_; i++)
    inc += test_errors_.row(i).has_inf();
  if (print)
    LOG(name_ +":"+std::to_string(size_t(100-O(inc/repeat_)))+"% completed.");
  if ( inc/repeat_ != 0 )
    return false;
  else 
    return true;
}
//-----------------------------------------------------------------------------     
// LCurve::_SplitData
//-----------------------------------------------------------------------------     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
std::vector<std::pair<arma::uvec,arma::uvec>>
  LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SplitData ( const DATASET& dataset,
                                                   const size_t seed )
{
  
  std::vector<std::pair<arma::uvec,arma::uvec>> data;
    split_(dataset.size_,Ns_,size_t(1),data,seed);
  return data;
}
//-----------------------------------------------------------------------------     
// LCurve::_RegisterSignalHandler
//-----------------------------------------------------------------------------     
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
  // this handles termination
  signal(SIGTERM, LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SignalHandler);
  // this handles kill
  signal(SIGKILL, LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_SignalHandler);

}
//-----------------------------------------------------------------------------     
// LCurve::_SignalHandler
//-----------------------------------------------------------------------------     
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
  LOG("Stopping program for some reason! Exiting..." << std::flush);
  if ( sig == SIGINT || sig == SIGTERM || sig == SIGKILL )
    std::quick_exit(0);
}

//-----------------------------------------------------------------------------     
// LCurve::_CleanUp
//-----------------------------------------------------------------------------     
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
void LCurve<MODEL,DATASET,SPLIT,LOSS,O>::_CleanUp ( )
{
  LOG("CleanUp is called!"<<std::flush);
  Save( name_ );
}
//-----------------------------------------------------------------------------     
// LCurve::Save
//-----------------------------------------------------------------------------     
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
  LOG("LCurve object saved to " << (path_/filename) << std::flush);
}
//-----------------------------------------------------------------------------     
// LCurve::Load
//-----------------------------------------------------------------------------     
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
  LOG("LCurve loaded from " << filename);
  return lcurve;
}
//-----------------------------------------------------------------------------     
// LCurve::_GetHPT
//-----------------------------------------------------------------------------     
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
//-----------------------------------------------------------------------------
// LCurve::_GetModel
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
// LCurve::GetName
//-----------------------------------------------------------------------------
template<class MODEL,
         class DATASET,
         class SPLIT,
         class LOSS,
         class O>
std::string LCurve<MODEL,DATASET,SPLIT,LOSS,O>::GetName ( )
{
  return name_;
}

} // namespace lcurve


#endif
