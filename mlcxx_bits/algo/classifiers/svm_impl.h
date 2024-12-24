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
 
template<class KERNEL,size_t SOLVER,class T>
template<class... Args>
SVM<KERNEL,SOLVER,T>::SVM ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels,
                     const size_t& num_class,
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

template<class KERNEL,size_t SOLVER,class T>
template<class... Args>
SVM<KERNEL,SOLVER,T>::SVM ( const arma::Mat<T>& inputs,
                            const arma::Row<size_t>& labels,
                            const size_t& num_class,
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

template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::Train ( const arma::Mat<T>& X,
                                   const arma::Row<size_t>& y )
{
  if (solver_ == "QP")
    _QP(X,y);
  else if (solver_ == "randSMO")
    _randSMO(X,y);
  else if (solver_ == "fanSMO")
    _fanSMO(X,y);
  else
    ERR("Not Implemented: Try QP or SMO");
}

template<class KERNEL,size_t SOLVER,class T>
std::pair<int,int> SVM<KERNEL,SOLVER,T>::_selectset ( arma::Row<T> G,
                                                      arma::Mat<T> Q )
{
  size_t len = y_.n_elem;

  // Step 1: Select `i`
  arma::uvec indices_i = arma::find(((y_ == +1) % (alphas_ < C_)) 
      || ((y_ == -1) % (alphas_ > 0)));
  arma::Col<T> G_candidates_i = -y_(indices_i) % G(indices_i);
  
  // Find the index of the maximum element in G_candidates_i
  int i = -1;
  T G_max = -arma::datum::inf;
  if (!G_candidates_i.is_empty()) 
  {
    arma::uword i_max_idx;
    G_max = G_candidates_i.max(i_max_idx);
    i = indices_i(i_max_idx);
  }

  // Step 2: Select `j`
  arma::uvec indices_j = arma::find(((y_ == +1) % (alphas_ > 0.)) ||
                                    ((y_ == -1) % (alphas_ < C_)));
  int j = -1;
  T obj_min = arma::datum::inf;
  T G_min = arma::datum::inf;

  #pragma omp parallel for
  for (size_t t=0; t<len; t++) {
      T b = G_max + y_[t] * G[t];
      
      if (-y_[t] * G[t] <= G_min) {
          G_min = -y_[t] * G[t];
      }

      if (b > 0) {
          T a = Q(i, i) + Q(t, t) - 2 * y_[i] * y_[t] * Q(i, t);
          if (a <= 0) a = tau_;
          T obj_val = -(b * b) / a;
          if (obj_val <= obj_min) {
              j = t;
              obj_min = obj_val;
          }
      }
  }

  // Check stopping criterion
  if (G_max - G_min < eps_) {
      return {-1, -1};
  }

  return {i, j};
}

template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::_fanSMO ( const arma::Mat<T>& X,
                                     const arma::Row<size_t>& y )
{
  X_ = &X;
  y_  = (arma::conv_to<arma::Row<int>>::from((y==ulab_(0)) * -2 + 1));
  alphas_.resize(y_.n_elem);
  arma::Row<T> G(y_.n_elem); G.fill(-1);
  arma::Mat<T> K = cov_.GetMatrix(X,X);
  arma::Mat<T> Q = (y_.t() * y_) % K;

  while (max_iter_>iter_++) 
  {
    auto [i, j] = _selectset(G, Q);
    if (j == -1) break;  // Termination condition if no valid (i, j) is found

    // Compute `a` and set to tau if non-positive
    T a = Q(i, i) + Q(j, j) - 2 * y_[i] * y_[j] * Q(i, j);
    if (a <= 0) a = tau_;

    // Compute `b`
    T b = -y_[i] * G[i] + y_[j] * G[j];

    // Store old alpha values
    T oldAi = alphas_[i], oldAj = alphas_[j];

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
    G += (Q.col(i) * deltaAi + Q.col(j) * deltaAj).t();
  }
  idx_ = arma::find(alphas_ >= 0.); 
}

template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::_randSMO ( const arma::Mat<T>& X,
                               const arma::Row<size_t>& y )
{
  X_ = &X;
  y_  = (arma::conv_to<arma::Row<int>>::from((y==ulab_(0)) * -2 + 1));

  size_t N = X.n_cols;
  alphas_.zeros(N);
  old_alphas_.zeros(N);
  arma::Mat<T> Kij;
  
  while ( (iter_++ == 0 || iter_ < max_iter_ ) & 
          (arma::norm(old_alphas_-alphas_,2) < eps_) )
  {
    old_alphas_ = alphas_;
    size_t i;
    arma::Row<T> updates(2);

    #pragma omp parallel for
    for ( size_t j=0; j<N; j++)
    {
      i = _geti(j,N);
      Kij = cov_.GetMatrix(X_->col(i).eval()) 
        + cov_.GetMatrix(X_->col(j).eval()) 
        - 2. * cov_.GetMatrix(X_->col(i).eval(), X_->col(j).eval());
      if (Kij(0,0) == 0.)
        continue;

      updates(0) = alphas_(i); updates(1) = alphas_(j); 
      arma::Row<T> LH = _getLH(i,j,updates);
      
      T Ei = _E(i);
      T Ej = _E(j);

      alphas_(j) = updates(1) + (T(y_(j)) * (Ei-Ej))/Kij(0,0);
      alphas_(j) = std::max(alphas_(j),LH(0));
      alphas_(j) = std::min(alphas_(j),LH(1));

      alphas_(i) = (updates(0) + T(y_(i)*y_(j)) * (updates(1)-alphas_(j)));

      b_ = _b(_w()); 
      
    }
  }
  idx_ = arma::find(alphas_ >= 0.); 
}

template<class KERNEL,size_t SOLVER,class T>
int SVM<KERNEL,SOLVER,T>::_E ( size_t i )
{
  arma::Row<T> w = _w();
  return (arma::sign( w * X_->col(i) + _b(w) ).eval() - y_(i)).eval()(0,0);
}

template<class KERNEL,size_t SOLVER,class T>
arma::Row<T> SVM<KERNEL,SOLVER,T>::_w ( )
{
  return (alphas_ % y_) * X_->t() ;
}

template<class KERNEL,size_t SOLVER,class T>
T SVM<KERNEL,SOLVER,T>::_b ( const arma::Row<T>& w )
{
  b_ = arma::mean(y_ - (w * (*X_))); 
  return b_;
}

template<class KERNEL,size_t SOLVER,class T>
arma::Row<T> SVM<KERNEL,SOLVER,T>::_getLH ( size_t i, size_t j,
                                     const arma::Row<T>& updates )
{
  arma::Row<T> LH(2);
  if (y_(i) != y_(j))
  {
    LH(0) = std::max(T(0.),updates(1)-updates(0));
    LH(1) = std::min(C_,C_-updates(0)+updates(1));
  }
  else
  {
    LH(0) = std::max(T(0.),updates(0)+updates(1)-C_);
    LH(1) = std::min(C_,updates(0)+updates(1));
  }
  return LH;
}

template<class KERNEL,size_t SOLVER,class T>
size_t SVM<KERNEL,SOLVER,T>::_geti ( size_t j, size_t N )
{
  /* arma::Row<T> is = {1,2,0,1,0,3,4,0,1,2}; */
  /* return is(iterx_++); */
  size_t draw;
  for (size_t h=0; h < max_iter_; h++)
  {
    draw = arma::randi(arma::distr_param(0,N-1));
    if (draw != j)
      return draw;
  }
  return draw;
  ERR("Could not draw a different i...");
}

template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::_QP ( const arma::Mat<T>& X,
                                 const arma::Row<size_t>& y )
{
  /* ulab_ = arma::unique(y); */
  /* BOOST_ASSERT_MSG(ulab_.n_elem <= 2 && ulab_.n_elem > 0, */
  /*                 "SVM:Only binary labels, please!"); */

  /* if (ulab_.n_elem == 1) */
  /* { */
  /*   oneclass_ = true; */
  /*   return; */
  /* } */

  X_ = &X;

  y_  = (arma::conv_to<arma::Row<int>>::from((y==ulab_(0)) * -2 + 1));

  int n_samples = X.n_cols;
  alphas_.ones(n_samples);
  
  // Compute the kernel matrix
  
  auto K = cov_.GetMatrix(X,X);
  // Formulate the QP problem
  arma::Mat<T> P = K % ((y_).t() * (y_));

  arma::Row<T> q = -arma::ones<arma::Row<T>>(n_samples);
  arma::Mat<T> G = arma::join_cols(
                              -arma::eye<arma::Mat<T>>(n_samples, n_samples),
                              arma::eye<arma::Mat<T>>(n_samples, n_samples));
  arma::Row<T> h = arma::join_rows(arma::zeros<arma::Row<T>>(n_samples),
                                  arma::ones<arma::Row<T>>(n_samples) * C_);

  arma::Mat<T> A = arma::conv_to<arma::Mat<T>>::from(y_);
  arma::Row<T> b = arma::zeros<arma::Row<T>>(1);
  bool success = opt::quadprog(alphas_,P, q, G, h, A, b);
  if (!success)
    ERR("HiGHS did not terminate by itself!");

  idx_ = arma::find(alphas_ >= 0. && alphas_<=C_); // Find the support vectors
  b_ = arma::accu(arma::conv_to<arma::Row<T>>::from(y_.cols(idx_))
        - ((alphas_ % y_) * cov_.GetMatrix(X, X.cols(idx_))))/idx_.n_elem;
}

template<class KERNEL,size_t SOLVER,class T>
void SVM<KERNEL,SOLVER,T>::Classify ( const arma::Mat<T>& inputs,
                                      arma::Row<size_t>& preds ) 
{
  if (!oneclass_)
  {
    arma::Mat<T> temp;
    if (nclass_==2)
      Classify(inputs,preds,temp);
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
        arma::Row<T> w = _w();
        dec_func =  w * inputs + _b(w) ;
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

template<class KERNEL,size_t SOLVER,class T>
T SVM<KERNEL,SOLVER,T>::ComputeError ( const arma::Mat<T>& points, 
                                       const arma::Row<size_t>& responses ) 
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class KERNEL,size_t SOLVER,class T>
T SVM<KERNEL,SOLVER,T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                                          const arma::Row<size_t>& responses )
{
  return (1. - ComputeError(points, responses));
}

} // namespace classification
} // namespace algo
#endif
