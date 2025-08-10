/**
 * @file nn_impl.h
 * @author Ozgur Taylan Turan
 *
 * Simple neural network wrapper for using learning curve generation and 
 * hyper-parameter tuning.
 *
 */
#ifndef NN_IMPL_H
#define NN_IMPL_H

namespace algo {

//-----------------------------------------------------------------------------
// ANN
//-----------------------------------------------------------------------------
/* template<class NET,class OPT,class MET,class O> */
/* template<class... OptArgs> */
/* ANN<NET,OPT,MET,O>::ANN ( NET* network, bool early, const OptArgs&... args ) */ 
/* { */
/*   network_ = network; */
/*   early_ = early; */
/*   opt_ = std::make_unique<OPT>(args...); */
/* } */

template<class NET,class OPT,class MET,class O>
ANN<NET,OPT,MET,O>::ANN ( const arma::Mat<O>& inputs,
                          const arma::Mat<O>& labels ) 
{
  early_ = false;
  opt_ = std::make_unique<OPT>();
  Train(inputs,labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
ANN<NET,OPT,MET,O>::ANN ( const arma::Mat<O>& inputs,
                          const arma::Mat<O>& labels,
                          const NET network ) 

{
  network_ = network;
  early_ = false;
  opt_ = std::make_unique<OPT>();
  Train(inputs,labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
template<class... OptArgs>
ANN<NET,OPT,MET,O>::ANN ( const arma::Mat<O>& inputs,
                          const arma::Mat<O>& labels,
                          const NET network, bool early, const OptArgs&... args ) 

{
  network_ = network;
  early_ = early;
  opt_ = std::make_unique<OPT>(args...);
  Train(inputs,labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
template<class... OptArgs>
ANN<NET,OPT,MET,O>::ANN ( const arma::Mat<O>& inputs,
                          const arma::Row<size_t>& labels,
                          const NET network, bool early, const OptArgs&... args ) 

{
  network_ = network;
  early_ = early;
  opt_ = std::make_unique<OPT>(args...);
  Train(inputs,labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
template<class... OptArgs>
void ANN<NET,OPT,MET,O>::Train ( const arma::Mat<O>& inputs,
                                 const arma::Row<size_t>& labels,
                                 const NET network, bool early,
                                 const OptArgs&... args ) 

{
  network_ = network;
  early_ = early;
  opt_ = std::make_unique<OPT>(args...);
  Train(inputs,labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
template<class... OptArgs>
void ANN<NET,OPT,MET,O>::Train ( const arma::Mat<O>& inputs,
                                 const arma::Mat<O>& labels,
                                 const NET network, bool early,
                                 const OptArgs&... args ) 

{
  network_ = network;
  early_ = early;
  opt_ = std::make_unique<OPT>(args...);
  Train(inputs,labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
void ANN<NET,OPT,MET,O>::Train( const arma::Mat<O>& inputs,
                                const arma::Mat<O>& labels ) 
{
  // Safety Net for learning curve generation from scratch,
  // but you might want to start from a trained model. So future modification
  // might be needed...
  if (network_.Parameters().n_elem != 0)
    network_.Reset();
  if (!early_)
    network_.Train(inputs,labels,*opt_);
  else
  {
    arma::Mat<O> inp,lab,val_inp,val_lab;

    mlpack::data::Split(inputs,labels,
                        inp,val_inp,lab,val_lab,0.2);

    auto func = [&](const arma::Mat<O>& inputs )
    {
      arma::Mat<O> pred;
      network_.Predict(val_inp, pred);
      return MET::Evaluate(pred, val_lab)/pred.n_cols;
    };

    ens::EarlyStopAtMinLossType<arma::Mat<O>> stop(func,5);
    network_.Train(inp,lab,*opt_,stop);

  }
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
void ANN<NET,OPT,MET,O>::Train( const arma::Mat<O>& inputs,
                                const arma::Row<size_t>& labels ) 
{
  // Get the unique labels
  this -> ulab_ = arma::unique(labels);
  // OneHotEncode the labels for the classifier network
  auto convlabels = _OneHotEncode(labels,ulab_);

  // Safety Net for learning curve generation from scratch,
  // but you might want to start from a trained model. So future modification
  // might be needed...
  if (network_.Parameters().n_elem != 0)
    network_.Reset();
  if (!early_)
    network_.Train(inputs,convlabels,*opt_);
  else
  {
    arma::Mat<O> inp,lab,val_inp,val_lab;

    mlpack::data::Split(inputs,convlabels,
                        inp,val_inp,lab,val_lab,0.2);

    auto func = [&](const arma::Mat<O>& inputs )
    {
      arma::Mat<O> pred;
      network_.Predict(val_inp, pred);
      return MET::Evaluate(pred, val_lab)/pred.n_cols;
    };

    ens::EarlyStopAtMinLossType<arma::Mat<O>> stop(func,5);
    network_.Train(inp,lab,*opt_,stop);

  }
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
void ANN<NET,OPT,MET,O>::Predict( const arma::Mat<O>& inputs,
                                  arma::Mat<O>& preds )
{
  network_.Predict(inputs,preds);
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
void ANN<NET,OPT,MET,O>::Classify( const arma::Mat<O>& inputs,
                                   arma::Row<size_t>& preds )
{
  arma::Mat<O> temp;
  network_.Predict(inputs,temp);

  preds = _OneHotDecode(temp,ulab_);
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
O ANN<NET,OPT,MET,O>::ComputeError( const arma::Mat<O>& inputs,
                                    const arma::Mat<O>& labels )
{
  arma::Mat<O> preds;
  network_.Predict(inputs,preds);
  return MET::Evaluate(preds, labels)/preds.n_elem;
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
arma::Mat<O> ANN<NET,OPT,MET,O>::_OneHotEncode( 
                                const arma::Row<size_t>& labels,
                                const arma::Row<size_t>& ulabels )
{
  size_t i=0;
  return arma::Mat<O>(ulabels.n_elem, labels.n_elem).
    each_col( [&](arma::vec& col){col(ulabels[labels[i++]])=1.;} );
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
arma::Row<size_t> ANN<NET,OPT,MET,O>::_OneHotDecode( 
                                      const arma::Mat<O>& labels,
                                      const arma::Row<size_t>& ulabels )
{
  return ulabels.cols(arma::index_max(labels,0));
}
///////////////////////////////////////////////////////////////////////////////
template<class NET,class OPT,class MET,class O>
arma::Mat<O> ANN<NET,OPT,MET,O>::Parameters ( ) 
{
  return network_.Parameters();
}

} // namespace algo

#endif 

