/**
 * @file nn_impl.h
 * @author Ozgur Taylan Turan
 *
 * Simple Neural Network Wrapper
 *
 */
#ifndef NN_IMPL_H
#define NN_IMPL_H

namespace algo { 
namespace regression {

//=============================================================================
// ANN 
//=============================================================================
/* template<class Optimizer,class T> */
/* template<class... Args> */
/* ANN<Optimizer,T>::ANN ( const arma::Row<size_t>& layer_info, */
/*                         const size_t& nonlintype, */
/*                         const bool& early, */
/*                         const Args&... args ) */ 
/*                         : layer_info_(layer_info), nonlintype_(nonlintype), */
/*                           archtype_(0), early_(early), opt_(args...) */
/* { */
/*   arma::Row<size_t>::iterator it = layer_info_.begin(); */
/*   arma::Row<size_t>::iterator it_end = layer_info_.end(); */

/*   for (; it!=it_end-1; it++) */
/*   { */
/*     model_.Add<mlpack::LinearType<arma::Mat<T>>>(size_t(*it)); */
/*     switch (nonlintype_) */
/*     { */
/*       case 1: */
/*         model_.Add<mlpack::ReLUType<arma::Mat<T>>>(); */
/*         break; */
/*       case 2: */
/*         model_.Add<mlpack::HardSigmoidType<arma::Mat<T>>>(); */
/*         break; */
/*       case 3: */
/*         model_.Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::Mat<T>>>(); */
/*         break; */
/*       case 4: */
/*         model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::Mat<T>>>(); */
/*         break; */
/*       default: */
/*         model_.Add<mlpack::ReLUType<arma::Mat<T>>>(); */
/*     } */
/*   } */
/*   model_.Add<mlpack::LinearType<arma::Mat<T>>>(size_t(*it)); */
/* } */

/* template<class Optimizer,class T> */
/* template<class... Args> */
/* ANN<Optimizer,T>::ANN ( const arma::Mat<T>& inputs, */
/*                         const arma::Mat<T>& labels, */
/*                         const arma::Row<size_t>& layer_info, */
/*                         const size_t& nonlintype, */
/*                         const bool& early, */
/*                         const Args&... args ) */ 
/*                       : ANN<Optimizer>::ANN( layer_info, nonlintype, early, args...) */ 
/* { */
/*   Train(inputs, labels); */
/* } */

template<class Optimizer,class T>
template<class... Args>
ANN<Optimizer,T>::ANN ( const arma::Mat<T>& inputs,
                        const arma::Mat<T>& labels,
                        const Args&... args ) 
                    : nonlintype_(0),
                      archtype_(0), early_(true) 
{

  opt_ = std::make_unique<Optimizer>(args...);
  PRINT("INTRAINCONSTRUCTOR-ANN")
  switch (archtype_)
  {
    case 1:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
      break;
    case 2:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(10);
      switch (nonlintype_)
      {
        case 1:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
          break;
        case 2:
          model_.Add<mlpack::HardSigmoidType<arma::Mat<T>>>();
          break;
        case 3:
          model_.template Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::Mat<T>>>();
          break;
        case 4:
          model_.template Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::Mat<T>>>();
          break;
        default:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
      }
      break;
    case 3:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(10);
      switch (nonlintype_)
      {
        case 1:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
          break;
        case 2:
          model_.Add<mlpack::HardSigmoidType<arma::Mat<T>>>();
          break;
        case 3:
          model_.Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::Mat<T>>>();
          break;
        case 4:
          model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::Mat<T>>>();
          break;
        default:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
      }
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
      break;
    default:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
  }
  Train(inputs, labels);
}



template<class Optimizer,class T>
template<class... Args>
ANN<Optimizer,T>::ANN ( const arma::Mat<T>& inputs,
                        const arma::Mat<T>& labels,
                        const size_t& archtype,
                        const size_t& nonlintype,
                        const bool& early,
                        const Args&... args ) 
                    : nonlintype_(nonlintype),
                      archtype_(archtype), early_(early)
{
  opt_ = std::make_unique<Optimizer>(args...);
  switch (archtype_)
  {
    case 1:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
      break;
    case 2:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(10);
      switch (nonlintype_)
      {
        case 1:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
          break;
        case 2:
          model_.Add<mlpack::HardSigmoidType<arma::Mat<T>>>();
          break;
        case 3:
          model_.template Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::Mat<T>>>();
          break;
        case 4:
          model_.template Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::Mat<T>>>();
          break;
        default:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
      }
      break;
    case 3:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(10);
      switch (nonlintype_)
      {
        case 1:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
          break;
        case 2:
          model_.Add<mlpack::HardSigmoidType<arma::Mat<T>>>();
          break;
        case 3:
          model_.Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::Mat<T>>>();
          break;
        case 4:
          model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::Mat<T>>>();
          break;
        default:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
      }
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
      break;
    default:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
  }
  Train(inputs, labels);
}

template<class Optimizer,class T>
void ANN<Optimizer,T>::Train ( const arma::Mat<T>& inputs,
                               const arma::Mat<T>& labels )
{
  if (!early_)
    model_.Train(inputs, labels, *opt_);
  else
  {
    arma::Mat<T> inp,lab,val_inp,val_lab;

    // If there is not enough data for the split train-test is the same
    mlpack::data::Split(inputs,labels,
                        inp,val_inp,lab,val_lab,potentialsplitratio_);

    auto func = [&](const arma::Mat<T>& inputs )
    {
      arma::mat pred;
      model_.Predict(val_inp, pred);
      return MET::Evaluate(pred, val_lab) / pred.n_cols;
    };
    ens::EarlyStopAtMinLossType<arma::Mat<T>> stop(func, ptc_);
    model_.Train(inp,lab,*opt_,stop);
  }
}

template<class Optimizer,class T>
const arma::Mat<T>& ANN<Optimizer,T>::Parameters ( ) const
{
  return model_.Parameters();
}

template<class Optimizer,class T>
arma::Mat<T>& ANN<Optimizer,T>::Parameters ( ) 
{
  return model_.Parameters();
}


template<class Optimizer,class T>
void ANN<Optimizer,T>::Predict ( const arma::Mat<T>& inputs,
                                 arma::Mat<T>& labels )
{
  model_.Predict(inputs, labels);
}


} // regression
} // algo

#endif 
