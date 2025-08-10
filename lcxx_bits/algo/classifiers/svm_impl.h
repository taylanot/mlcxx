/**
 * @file svm.h
 * @author Ozgur Taylan Turan
 *
 * SVM Classfier
 *
 */

#ifndef SVM_IMPL_H
#define SVM_IMPL_H

namespace algo {
namespace classification {
//----------------------------------------------------------------------------- 
// SVM
//----------------------------------------------------------------------------- 
template<class KERNEL,size_t SOLVER,class T>
template<class... Args>
SVM<KERNEL,SOLVER,T>::SVM ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels,
                     const size_t num_class,
                     const T& C,
                     const Args&... args) : 
                      nclass_(num_class), C_(C), cov_(args...)
                                                        
{
  ulab_ = arma::unique(labels);

  if (ulab_.n_elem == 1)
  {
    oneclass_ = true;
    return;
  }

  else
  {
    if (nclass_ == 2)
    {
      this->Train(inputs,labels);
    }
    else
    {
      ova_ = OnevAll<SVM<KERNEL,SOLVER,T>>(inputs, labels,
                                    nclass_, size_t(2), C, args...);
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
template<class... Args>
SVM<KERNEL,SOLVER,T>::SVM ( const arma::Mat<T>& inputs,
                            const arma::Row<size_t>& labels,
                            const size_t num_class,
                            const Args&... args ) :
                      nclass_(num_class), C_(T(1.0)), cov_(args...)
{
  ulab_ = arma::unique(labels);

  if (ulab_.n_elem == 1)
  {
    oneclass_ = true;
    return;
  }
  else
  {
    if (nclass_ == 2)
      this->Train(inputs,labels);
    else
      ova_ = OnevAll<SVM<KERNEL,SOLVER,T>>(inputs, labels,
                                    nclass_, size_t(2),C_,args...);
  }
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::Train ( const arma::Mat<T>& X,
                                   const arma::Row<size_t>& y )
{
  if (solver_ == "fanSMO")
    _fanSMO(X,y);
  else
    ERR("Not Implemented: Try fanSMO");
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::Train ( const arma::Mat<T>& X,
                                   const arma::Row<size_t>& y,
                                   const size_t num_class )
{
 this -> nclass_ = num_class;
 this -> Train(X,y);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
std::pair<int,int> SVM<KERNEL,SOLVER,T>::_selectset ( arma::Row<T> G,
                                                      arma::Mat<T> Q )
{
  T inf =  arma::datum::inf;
  T G_max = -inf;
  T G_min = inf;
  T obj_min = inf;
  size_t len = y_.n_elem;

  int i = -1;
  for (size_t t=0; t<len; t++)
  {
    if ( (y_[t] == +1 && alphas_[t] < C_) || (y_[t] == -1 && alphas_[t] > 0.) )
    {
      if ( -y_[t]*G[t] >= G_max )
      {
        i = t;
        G_max = -y_[t] * G[t];
      }
    }
  }

  int j = -1;
  for (size_t t=0; t<len; t++)
  {
    if ( (y_[t] == +1 && alphas_[t] > 0.) || (y_[t] == -1 && alphas_[t] < C_) )
    {
      T b = G_max + y_[t]*G[t];
      if ( -y_[t]*G[t] <= G_max )
        G_min = -y_[t]*G[t];
      if ( b > 0. )
      {
        T a = Q(i,i) + Q(t,t) - 2.*y_[i]*y_[t]*Q(i,t);
        if ( a <= 0. )
          a = tau_;
        if (-(b*b)/a <= obj_min)
        {
          j = t; 
          obj_min = -(b*b)/a;
        }
      }

    }
  } 

  if (G_max-G_min < eps_)
    return {-1,-1};
  else
    return {i, j};
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::_fanSMO ( const arma::Mat<T>& X,
                                     const arma::Row<size_t>& y )
{
  X_ = X;
  y_  = (arma::conv_to<arma::Row<int>>::from((y==ulab_(0)) * -2 + 1));
  size_t N = y_.n_elem;
  alphas_.resize(N);
  /* alphas_ = 0.; */
  arma::Row<T> G(N); G.fill(-1.);
  arma::Mat<T> K;
  /* if (N > 200) */
  /*   K = cov_.GetMatrix_approx(X,X,200); */
  /* else */
  /*   K = cov_.GetMatrix(X,X); */
  K = cov_.GetMatrix(X,X);
  arma::Mat<T> Q = (y_.t() * y_) % K;
  while (max_iter_>iter_++) 
  {
    auto [i, j] = _selectset(G, Q);
    if (j == -1) break;  // Termination condition if no valid (i, j) is found

    // Compute `a` and set to tau if non-positive
    T a = Q(i, i) + Q(j, j) - 2 * y_[i]*y_[j] * Q(i, j);
    if (a <= 0.) 
      a = tau_;

    // Compute `b`
    T b = -y_[i] * G[i] + y_[j] * G[j];

    // Store old alpha values
    T oldAi = alphas_[i];
    T oldAj = alphas_[j];

    // Update alpha values for i and j
    alphas_[i] += y_[i] * b / a;
    alphas_[j] -= y_[j] * b / a;

    // Project alpha values back to the feasible region [0, C]
    T sum = y_[i] * oldAi + y_[j] * oldAj;

    // Project A[i] back to [0, C]
    alphas_[i] = std::clamp(alphas_[i], T(0.0), C_);
    
    // Adjust A[j] based on updated A[i] and maintain feasibility constraint
    alphas_[j] = y_[j] * (sum - y_[i] * alphas_[i]);
    alphas_[j] = std::clamp(alphas_[j], T(0.0), C_);

    // Re-adjust A[i] based on adjusted A[j]
    alphas_[i] = y_[i] * (sum - y_[j] * alphas_[j]);

    // Compute changes in alpha
    T deltaAi = alphas_[i] - oldAi;
    T deltaAj = alphas_[j] - oldAj;

    // Vectorized gradient update: G += Q.col(i) * deltaAi + Q.col(j) * deltaAj
    /* G += (Q.col(i) * deltaAi + Q.col(j) * deltaAj).t(); */
    size_t len = y_.n_elem;
    for (size_t h=0; h<len; h++)
    {
      G[h] += Q(h,i)*deltaAi+Q(h,j)*deltaAj;
    }
  }
  idx_ = arma::find(alphas_ > tau_); 
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::Classify ( const arma::Mat<T>& inputs,
                                      arma::Row<size_t>& preds ) 
{
  if (!oneclass_)
  {
    arma::Mat<T> temp;
    if (nclass_==2)
    {
      Classify(inputs,preds,temp);
    }
    else 
    {
      ova_.Classify(inputs,preds,temp);
    }
  }
  else 
  {
    preds.resize(inputs.n_cols);
    preds.fill(ulab_[0]);
  }
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::Classify ( const arma::Mat<T>& inputs,
                                      arma::Row<size_t>& preds,
                                      arma::Mat<T>& probs ) 
{
  arma::Mat<T> dec_func;

  if (!oneclass_)
  {
    if (nclass_==2)
    { 
      if (idx_.n_elem>0)
      {
        probs.set_size(nclass_,inputs.n_cols);
        preds.set_size(inputs.n_cols);
        arma::Mat<T> svs = X_.cols(idx_);
        arma::Mat<T> K = cov_.GetMatrix(svs,inputs);
        arma::Mat<T> Ksv = cov_.GetMatrix(svs);

        b_ = arma::accu(arma::conv_to<arma::Row<T>>::from(y_.cols(idx_))
        - ((alphas_.cols(idx_) % y_.cols(idx_)) * Ksv)) /idx_.n_elem;

        dec_func = (alphas_.cols(idx_) % y_.cols(idx_)) * K ;

        preds.elem( arma::find( dec_func <= 0.) ).fill(ulab_[0]);
        preds.elem( arma::find( dec_func > 0.) ).fill(ulab_[1]);
        probs.row(0) = 1. / (1. + arma::exp(dec_func));
        probs.row(1) = 1 - probs.row(0);
      }
      else
      {
          ERR("No support vectors->No prediction");
          return;
      }
    }
    else
      ova_.Classify(inputs, preds, probs);
  }
  else
  {
    probs.resize(nclass_,inputs.n_cols);
    probs.row(ulab_[0]).fill(1.);
    preds.resize(inputs.n_cols);
    preds.fill(ulab_[0]);
  }
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
T SVM<KERNEL,SOLVER,T>::ComputeError ( const arma::Mat<T>& points, 
                                       const arma::Row<size_t>& responses ) 
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,size_t SOLVER,class T>
T SVM<KERNEL,SOLVER,T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                                          const arma::Row<size_t>& responses )
{
  return (1. - ComputeError(points, responses));
}
///////////////////////////////////////////////////////////////////////////////
} // namespace classification
} // namespace algo
#endif
