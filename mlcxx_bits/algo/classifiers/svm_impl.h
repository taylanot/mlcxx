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
 
template<class KERNEL,class T>
template<class... Args>
SVM<KERNEL,T>::SVM ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels,
                     const size_t& num_class,
                     const double& C,
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
      ova_ = OnevAll<SVM<KERNEL,T>>(inputs, labels, size_t(2), C, args...);
    }
  }
}

template<class KERNEL,class T>
template<class... Args>
SVM<KERNEL,T>::SVM ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels,
                     const size_t& num_class,
                     const Args&... args ) :
                      nclass_(num_class), C_(1.0), cov_(args...)
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
      ova_ = OnevAll<SVM<KERNEL,T>>(inputs, labels, 2, C_, args...);
  }
}

template<class KERNEL,class T>
void SVM<KERNEL,T>::Train ( const arma::Mat<T>& X,
                            const arma::Row<size_t>& y )
{
  if (solver_ == "QP")
    _QP(X,y);
  else if (solver_ == "SMO")
    _SMO(X,y);
  else
    ERR("Not Implemented: Try QP or SMO");
}

template<class KERNEL,class T>
void SVM<KERNEL,T>::_SMO ( const arma::Mat<T>& X,
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
    for ( size_t j=0; j<N; j++)
    {
      i = _geti(j,N);
      Kij = cov_.GetMatrix(X_->col(i).eval()) + cov_.GetMatrix(X_->col(j).eval()) 
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

template<class KERNEL,class T>
int SVM<KERNEL,T>::_E ( size_t i )
{
  arma::Row<T> w = _w();
  return (arma::sign( w * X_->col(i) + _b(w) ).eval() - y_(i)).eval()(0,0);
}

template<class KERNEL,class T>
arma::Row<T> SVM<KERNEL,T>::_w ( )
{
  return (alphas_ % y_) * X_->t() ;
}

template<class KERNEL,class T>
T SVM<KERNEL,T>::_b ( const arma::Row<T>& w )
{
  b_ = arma::mean(y_ - (w * (*X_))); 
  return b_;
}

template<class KERNEL,class T>
arma::Row<T> SVM<KERNEL,T>::_getLH ( size_t i, size_t j,
                                     const arma::Row<T>& updates )
{
  arma::Row<T> LH(2);
  if (y_(i) != y_(j))
  {
    LH(0) = std::max(0.,updates(1)-updates(0));
    LH(1) = std::min(C_,C_-updates(0)+updates(1));
  }
  else
  {
    LH(0) = std::max(0.,updates(0)+updates(1)-C_);
    LH(1) = std::min(C_,updates(0)+updates(1));
  }
  return LH;
}

template<class KERNEL,class T>
size_t SVM<KERNEL,T>::_geti ( size_t j, size_t N )
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

template<class KERNEL,class T>
void SVM<KERNEL,T>::_QP ( const arma::Mat<T>& X,
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

template<class KERNEL,class T>
void SVM<KERNEL,T>::Classify ( const arma::Mat<T>& inputs,
                               arma::Row<size_t>& preds ) 
{
  if (!oneclass_)
  {
    arma::Row<T> temp;
    if (nclass_==2)
      Classify(inputs,preds,temp);
    else 
      ova_.Classify(inputs,preds);
  }
  else 
  {
    preds.resize(inputs.n_cols);
    preds.fill(ulab_[0]);
  }
}

template<class KERNEL,class T>
void SVM<KERNEL,T>::Classify ( const arma::Mat<T>& inputs,
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
        /* probs = (dec_func - arma::min(dec_func,1).eval()(0,0))  / (arma::max(dec_func,1) - arma::min(dec_func,1)).eval()(0,0); */
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
    dec_func.resize(inputs.n_cols);
    dec_func.fill(arma::datum::nan);
    preds.resize(inputs.n_cols);
    preds.fill(ulab_[0]);
  }
}

template<class KERNEL,class T>
T SVM<KERNEL,T>::ComputeError ( const arma::Mat<T>& points, 
                                const arma::Row<size_t>& responses ) 
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class KERNEL,class T>
T SVM<KERNEL,T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                                         const arma::Row<size_t>& responses )
{
  return (1. - ComputeError(points, responses))*100;
}

} // namespace classification
} // namespace algo
#endif
