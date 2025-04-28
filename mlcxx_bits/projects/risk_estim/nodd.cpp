/**
 * @file nodd.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for showing double descent mitigation via excluisoin of principle components.
 *
 */
#define DTYPE double  

#include <headers.h>



/* arma::Row<DTYPE> eig_vals(100*29); */
/* size_t counter = 0; */

struct solve_
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    /* return (b*arma::pinv(A,1)).t() ; */

    return arma::solve(A,b.t(),arma::solve_opts::allow_ugly);
    /* return arma::solve(A,b.t()); */
  }
};


struct hightol
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    return arma::pinv(A,0.1)*b.t();
  }
};

struct lowtol
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    return arma::pinv(A,1.e-6)*b.t();
  }
};

struct smarttol
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    arma::Mat<T> U,V;
    arma::Col<T> S;
    arma::svd_econ(U,S,V,A);
    return arma::pinv(A,T(S[0]*0.001))*b.t();
  }
};

struct pinv
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    return arma::pinv(A)*b.t();
  }
};

struct mypinv
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b, T tol) 
  {
    arma::Mat<T> U,V;
    arma::Col<T> s;
    arma::svd(U,s,V,A);
    arma::Col<T> s2(arma::size(s));

    int counter=0;
    for(size_t i=0; i < s.n_elem; ++i)
    {
      if(s[i]>= tol)  { s2[i] = (s[i] > T(0)) ? T(T(1) / s[i]) : T(0); counter++; }
    }
    return (V*arma::diagmat(s2)*U.t())*b.t();
  }


  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    T tol = arma::datum::eps;
    arma::Mat<T> U,V;
    arma::Col<T> s;
    arma::svd_econ(U,s,V,A);
    tol *= arma::max(s);
    arma::uvec idx = arma::find(s>tol);
    s.elem(idx) = T(1.)/s.elem(idx);
    idx = arma::find(s<=tol);
    s.elem(idx).zeros();
    return (V*arma::diagmat(s)*U.t())*b.t();
  }
};

struct rankpinv 
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    const size_t rank = arma::rank(A);
    if (rank < A.n_cols)
    {
      arma::Mat<T> U,V;
      arma::Col<T> s;
      arma::svd(U,s,V,A);
      arma::Col<T> s2(arma::size(s));
      const T rcond = arma::rcond(A);
      size_t nval;
      for(size_t i=0; i<rank; ++i)
       s2[i] = 1./s[i]; 
      return (V*arma::diagmat(s2)*U.t())*b.t();
    }
    else
    {
    return arma::pinv(A)*b.t();
    }
  }
};

/* struct rankpinv */ 
/* { */
/*   template<class T=DTYPE> */
/*   arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) */ 
/*   { */
/*     arma::Mat<T> U,V; */
/*     arma::Col<T> s; */
/*     arma::svd(U,s,V,A); */
/*     arma::Col<T> s2(arma::size(s)); */
/*     const size_t rank = arma::rank(A); */
/*     const T rcond = arma::rcond(A); */

/*     if (rank != s.n_elem  || rcond < 1e-6) */
/*     { */
/*        s2[0] = 1./s[0]; */ 
/*     } */
/*     else */
/*     { */
/*       for(size_t i=0; i<rank; ++i) */
/*        s2[i] = 1./s[i]; */ 
/*     } */
/*     return (V*arma::diagmat(s2)*U.t())*b.t(); */
/*   } */
/* }; */


struct smartpinv 
{
  /* template<class T=DTYPE> */
  /* arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) */ 
  /* { */
  /*   T tol = 0.95; */
  /*   arma::Mat<T> U,V; */
  /*   arma::Col<T> s; */
  /*   arma::svd(U,s,V,A); */
  /*   T sum = arma::accu(s); */

  /*   arma::Col<T> s2(arma::size(s)); */

  /*   int summer=0; */
  /*   for(size_t i=0; i < s.n_elem; ++i) */
  /*   { */
  /*     if(summer <= tol) */  
  /*     { */
  /*       summer += s[i]/sum; */
  /*       s2[i] = (s[i] > T(0)) ? T(T(1) / s[i]) : T(0); */
  /*     } */
  /*   } */
  /*   return (V*arma::diagmat(s2)*U.t())*b.t(); */
  /* } */

  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    T tol = 10.;
    arma::Mat<T> U,V;
    arma::Col<T> s;
    arma::svd(U,s,V,A);
    arma::Col<T> s2(arma::size(s));
    size_t rank = arma::rank(A);

    for(size_t i=0; i < s.n_elem; ++i)
    {
      if ( rank < A.n_cols )
      {
        s2[i] = (i < rank-1 ) ? T(T(1) / s[i]) : T(0);
      }
      else if ( s[i] < tol )
      {
        s2[i] = (s[i] > tol) ? T(T(1) / s[i]) : T(0);
      }
      else
      {
        s2[i] = (i < rank ) ? T(T(1) / s[i]) : T(0);
      }

    }
    return (V*arma::diagmat(s2)*U.t())*b.t();
  }
};

struct mypinv2
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    if ( arma::rank(A) < A.n_rows )
    {
      arma::Mat<T> U,V;
      arma::Col<T> s;
      arma::svd(U,s,V,A);
      arma::uvec idx = arma::regspace<arma::uvec>(0,1,arma::rank(A)-1);
      s.elem(idx) = 1./s.elem(idx);
      /* idx = arma::find(s<=tol); */
      /* s.elem(idx).fill(0.); */
      return (V*arma::diagmat(s)*U.t())*b.t();
    }
    else
    {
      return arma::pinv(A)*b.t();
    }
  }
};

struct trunc_svd
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    if ( arma::rank(A) < A.n_rows )
    {
      size_t k = 1;
      arma::Mat<T> U,V;
      arma::Col<T> s;
      arma::svd(U,s,V,A);
      arma::uvec idx = arma::regspace<arma::uvec>(0,1,k); 
      
      return arma::pinv(V.cols(idx)*arma::diagmat(s.elem(idx))*U.cols(idx).t())*b.t();
    }
    else
    {
      return arma::pinv(A)*b.t();
    }
  }
};

struct minnorm
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    if ( arma::rank(A) < A.n_rows )
    {
      arma::Row<T> x;
      size_t d = A.n_rows;
      arma::Mat<T> Q = arma::eye<arma::Mat<T>>(d,d);
      arma::Row<T> c(d);

      arma::Mat<T> G;
      arma::Row<T> h;

      opt::quadprog(x,Q,c,G,h,A,b,false,false);
      return x.t();
    }

    else
    {
      return arma::solve(A,b.t());
    }

  }
};

template<class SOLVER=pinv,class T=DTYPE>
class LinearRegression
{
public:

  LinearRegression (T lambda, bool bias) : 
                                              bias_(bias), lambda_(lambda) { }

  template<class... Args>
  LinearRegression ( const arma::Mat<T>& X,const arma::Mat<T>& y,
                     T lambda, bool bias)
  : bias_(bias), lambda_(lambda)
  {
    Train(X,y);
  }

  void Train( const arma::Mat<T>& X, const arma::Mat<T>& y)
  {
    arma::Mat<T> X_;
    if (bias_)
      X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols));
    else 
      X_ = X;

    auto b = arma::conv_to<arma::Row<T>>::from(X_ * y.t());

    parameters_ = solver_(
      ((X_*X_.t())+(arma::eye<arma::Mat<T>>(X_.n_rows,X_.n_rows)*lambda_)).eval(),b);
  }

  void Predict( const arma::Mat<T>& X, arma::Mat<T>& preds)
  {
    arma::Mat<T> X_;
    if (bias_)
      X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols));
    else
      X_ = X;
    preds = parameters_.t() * X_;
  }

  arma::Mat<T> Parameters( ){return parameters_.t();}

  T ComputeError( const arma::Mat<T>& X, const arma::Mat<T>& preds)
  {
    arma::Mat<T> temp;
    Predict(X,temp);
    return arma::dot(temp,temp)/temp.n_elem;
    
  }
  
private:
  arma::Mat<T> parameters_;
  SOLVER solver_;
  bool bias_;
  T lambda_ ;
};

template<class SOLVER=pinv,class T=DTYPE>
class LinearRegressionPCA
{
public:

  LinearRegressionPCA (T lambda, bool bias) : 
                                              bias_(bias), lambda_(lambda) { }

  template<class... Args>
  LinearRegressionPCA ( const arma::Mat<T>& X,const arma::Mat<T>& y,
                     T lambda, bool bias)
  : bias_(bias), lambda_(lambda)
  {
    arma::Mat<T> X_;
    mean_ = arma::mean(X,1);
    arma::Col<T> vals, tsqr;
    arma::princomp(vecs_,X_,vals,tsqr,X.t());
    Train(X_.t(),y);
  }

  void Train( const arma::Mat<T>& X, const arma::Mat<T>& y)
  {
    arma::Mat<T> X_;
    if (bias_)
      X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols));
    else 
      X_ = X;

    auto b = arma::conv_to<arma::Row<T>>::from(X_ * y.t());

    parameters_ = solver_(
      ((X_*X_.t())+(arma::eye<arma::Mat<T>>(X_.n_rows,X_.n_rows)*lambda_)).eval(),b);
  }

  void Predict( const arma::Mat<T>& X, arma::Mat<T>& preds)
  {
    arma::Mat<T> X_ = vecs_*(X.each_col() - mean_);
    if (bias_)
      X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols));
    preds = parameters_.t() * X_;
  }

  arma::Mat<T> Parameters( ){return parameters_.t();}

  T ComputeError( const arma::Mat<T>& X, const arma::Mat<T>& preds)
  {
    arma::Mat<T> temp;
    Predict(X,temp);
    return arma::dot(temp,temp)/temp.n_elem;
    
  }
  
private:
  arma::Mat<T> parameters_;
  SOLVER solver_;
  bool bias_;
  T lambda_ ;
  arma::Mat<T> vecs_;
  arma::Mat<T> mean_;
};
template<class T=DTYPE>
class EigVal
{
public:

  EigVal ( const arma::Mat<T>& X,const arma::Mat<T>& y, const T stddev ) : std_(stddev) 
  {
    Train(X,y);
  }

  void Train( const arma::Mat<T>& X, const arma::Mat<T>& y)
  {
    arma::Mat<T> U,V;
    arma::Col<T> S;
    arma::Mat<T> A = (X*X.t());
    arma::svd(U,S,V,A);
    arma::Row<DTYPE> s2(S.n_elem);
    D_ = S.n_elem;

    int counter=0;
    for(size_t i=0; i < S.n_elem; ++i)
    {
      s2[i] = (i < arma::rank(A)) ? DTYPE(DTYPE(1.) / (S[i]*S[i])) : DTYPE(0);
      counter++;
    }
    parameters_ = s2;
  }

  void Predict( const arma::Mat<T>& X, arma::Mat<T>& preds)
  {
    preds.resize(1,1);
    preds[0] = std_*std_*(arma::accu(parameters_)+1.);
    /* preds[0] = arma::accu(parameters_); */
  }

  arma::Row<T> Parameters( ){return parameters_;}

private:
  arma::Row<T> parameters_;
  T std_;
  size_t D_;
};

int main ( int argc, char** argv )
{
  /* std::filesystem::path path = EXP_PATH/"15_08_23/smart20D"; */
  /* std::filesystem::path path = EXP_PATH/"19_08_23/eigval2"; */
  /* std::filesystem::path path = EXP_PATH/"19_08_23/comparison"; */
  std::filesystem::path path = EXP_PATH/"30_08_23/dist2";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();

  /* arma::irowvec Ds = {5,10,20}; */
  arma::irowvec Ds = {5};
  DTYPE stddev = 0.1;

  size_t rep = 100;

  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(2,10,100); */
  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,20);

  /* arma::Row<DTYPE> a = {100,6,3,4,5}; */
  /* arma::Row<DTYPE> a(5,arma::fill::ones); */
  /* mlpack::RandomSeed(10); */
  for (size_t i=0;i<Ds.n_elem;i++)
  {

    arma::Col<DTYPE> mean = arma::zeros<arma::Col<DTYPE>>(Ds[i]);
    arma::Mat<DTYPE> cov =  arma::eye(Ds[i],Ds[i]);
    cov(1,1) = 100000;
    /* data::regression::Dataset dataset(Ds(i),10000); */
    data::regression::Dataset dataset(Ds(i),10000,mean,cov);
    /* data::regression::Dataset dataset(Ds(i),4); */

    dataset.Generate(std::string("Linear"),stddev);
    /* dataset.Generate(a,stddev); */

    /* { */
    /*   /1* auto X = arma::join_vert(dataset.inputs_,arma::ones(1,dataset.inputs_.n_cols)); *1/ */
    /*   arma::Mat<DTYPE> X = dataset.inputs_; */

    /*   arma::Mat<DTYPE> A = X * X.t(); */
    /*   arma::Col<DTYPE> b = (X*dataset.labels_.t()); */

    /*   arma::Col<DTYPE> def = arma::pinv(A)*b; */
    /*   arma::Col<DTYPE> defreg = arma::pinv(A+arma::eye(arma::size(A))*Ds[i]*stddev)*b; */
    /*   arma::Col<DTYPE> lowtol = arma::pinv(A,1.e-17)*b; */
    /*   arma::Col<DTYPE> hightol = arma::pinv(A,0.001)*b; */

    /*   minnorm minnorm_; */
    /*   mypinv  mypinv_; */
    /*   smarttol  smarttol_; */
    /*   rankpinv ranknorm_; */
    /*   arma::Col<DTYPE> mnorm = minnorm_(A,arma::Row<DTYPE>(b.t())); */
    /*   arma::Col<DTYPE> myp = mypinv_(A,arma::Row<DTYPE>(b.t()),1.); */
    /*   arma::Col<DTYPE> smart = smarttol_(A,arma::Row<DTYPE>(b.t())); */
    /*   arma::Col<DTYPE> rank = ranknorm_(A,arma::Row<DTYPE>(b.t())); */

    /*   PRINT_VAR(rank); */
    /*   PRINT_VAR(arma::norm(rank,2)); */

    /*   PRINT_VAR(def); */
    /*   PRINT_VAR(arma::norm(def,2)); */

    /*   PRINT_VAR(mnorm); */
    /*   PRINT_VAR(arma::norm(mnorm,2)); */

    /*   PRINT_VAR(defreg); */
    /*   PRINT_VAR(arma::norm(defreg,2)); */
    /*   PRINT_VAR(hightol); */
    /*   PRINT_VAR(arma::norm(hightol,2)); */
    /*   PRINT_VAR(lowtol); */
    /*   PRINT_VAR(arma::norm(lowtol,2)); */
      
    /*   PRINT_VAR(myp); */
    /*   PRINT_VAR(arma::norm(myp,2)); */
    /*   PRINT_VAR(smart); */
    /*   PRINT_VAR(arma::norm(smart,2)); */
    /*   PRINT_VAR(b); */
    /*   PRINT_VAR(A*smart); */
    /*   PRINT_VAR(A*mnorm); */
    /*   PRINT_VAR(A*lowtol); */
    /* } */
    
    /* LinearRegression<pinv> pinvmodel(dataset.inputs_,dataset.labels_); */
    /* LinearRegression<mypinv> mypinvmodel(dataset.inputs_,dataset.labels_); */
    /* LinearRegression<solve_> solvemodel(dataset.inputs_,dataset.labels_); */
    /* LinearRegression<minnorm> minnormmodel(dataset.inputs_,dataset.labels_); */

    /* PRINT_VAR(pinvmodel.Parameters()); */
    /* PRINT_VAR(mypinvmodel.Parameters()); */
    /* PRINT_VAR(solvemodel.Parameters()); */
    /* PRINT_VAR(minnormmodel.Parameters()); */
    /* PRINT_VAR(arma::norm(pinvmodel.Parameters(),2)); */
    /* PRINT_VAR(arma::norm(mypinvmodel.Parameters(),2)); */
    /* PRINT_VAR(arma::norm(solvemodel.Parameters(),2)); */
    /* PRINT_VAR(arma::norm(minnormmodel.Parameters(),2)); */

    /* { */
    /*   arma::Mat<DTYPE> eigs(rep,Ns.n_elem); */

    /*   ProgressBar prog(rep*Ns.n_elem-1); */
    /*   for (size_t j=0;j<Ns.n_elem-1;j++) */
    /*     for (size_t k=0;k<rep;k++) */
    /*     { */
    /*       auto G  = dataset.inputs_*dataset.inputs_.t(); */
    /*       arma::uvec idx = arma::randperm(dataset.size_,Ns(i)); */
    /*       arma::Mat<DTYPE> U,V; */
    /*       arma::Col<DTYPE> s; */
    /*       arma::svd(U,s,V,G); */
    /*       eigs(k,j) = s(s.n_elem-1); */
    /*       prog.Update(); */
    /*     } */
    /*   eigs.save("eigs_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* { */
    /*   dataset.inputs_*=2; */
    /*   arma::Mat<DTYPE> eigs(rep,Ns.n_elem); */
    /*   ProgressBar prog(rep*Ns.n_elem-1); */
    /*   for (size_t j=0;j<Ns.n_elem-1;j++) */
    /*     for (size_t k=0;k<rep;k++) */
    /*     { */
    /*       auto G  = dataset.inputs_*dataset.inputs_.t(); */
    /*       arma::uvec idx = arma::randperm(dataset.size_,Ns(i)); */
    /*       arma::Mat<DTYPE> U,V; */
    /*       arma::Col<DTYPE> s; */
    /*       arma::svd(U,s,V,G); */
    /*       eigs(k,j) = s(s.n_elem-1); */
    /*       prog.Update(); */
    /*     } */
    /*   eigs.save("scaled_eigs_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* { */
    /*   dataset.inputs_*=1e36; */
    /*   src::LCurve<LinearRegression<pinv>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,false); */
    /*   lc.test_errors_.save("pinv_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* { */
    /*   src::LCurve<LinearRegression<pinv2>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,true); */
    /*   lc.test_errors_.save("pinv2_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* { */
    /*   src::LCurve<LinearRegression<mypinv2>,mlpack::MSE> lc_(Ns,rep,false,false,true); */
    /*   lc_.Bootstrap(dataset.inputs_,dataset.labels_,false); */
    /*   lc_.test_errors_.save("mypinv_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* { */
    /*   dataset.inputs_+=400.; */
    /*   src::LCurve<LinearRegression<pinv2>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,true); */
    /*   lc.test_errors_.save("add_pinv_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */
      /* eig_vals.save("eigs_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */

    /* { */
    /*   src::LCurve<mlpack::LinearRegression<>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   /1* src::LCurve<mlpack::LinearRegression<>,mlpack::MSE> lc2(Ns,rep,true,false,true); *1/ */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_); */
    /*   lc.test_errors_.save("biggernorm_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /*   /1* dataset.inputs_+=100.; *1/ */
    /*   /1* dataset.labels_+=100.; *1/ */
    /*   /1* lc2.Bootstrap(dataset.inputs_,dataset.labels_); *1/ */
    /*   /1* lc2.test_errors_.save("add_pinv2_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); *1/ */
    /* } */

    //{
    //  src::LCurve<LinearRegression<pinv>,mlpack::MSE> lc(Ns,rep,true,false,true);
    //  lc.Bootstrap(dataset.inputs_,dataset.labels_,false,false);
    //  lc.test_errors_.save("svd_"+std::to_string(Ds(i))+".csv",arma::csv_ascii);
    //}
    
    {
      mlpack::RandomSeed(SEED);
      src::LCurve<LinearRegression<pinv>,mlpack::MSE> lc(Ns,rep,true,false,true);
      lc.Bootstrap(dataset.inputs_,dataset.labels_,0.,false);
      lc.test_errors_.save("def_"+std::to_string(Ds(i))+".csv",arma::csv_ascii);
    }

    /* { */
    /*   mlpack::RandomSeed(SEED); */
    /*   src::LCurve<LinearRegression<lowtol>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,0.,false); */
    /*   lc.test_errors_.save("lowtol_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* { */
    /*   mlpack::RandomSeed(SEED); */
    /*   src::LCurve<LinearRegression<hightol>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,0.,false); */
    /*   lc.test_errors_.save("hightol_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */


    /* { */
    /*   mlpack::RandomSeed(SEED); */
    /*   src::LCurve<LinearRegression<rankpinv>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,0.,false); */
    /*   lc.test_errors_.save("rankpinv_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* { */
    /*   mlpack::RandomSeed(SEED); */
    /*   src::LCurve<LinearRegression<smarttol>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,0.,false); */
    /*   lc.test_errors_.save("smarttol_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* { */
    /*   mlpack::RandomSeed(SEED); */
    /*   src::LCurve<LinearRegressionPCA<pinv>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,0.,false); */
    /*   lc.test_errors_.save("pca_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* mlpack::RandomSeed(SEED); */
    /* { */
    /*   src::LCurve<LinearRegression<minnorm>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,0.,false); */
    /*   lc.test_errors_.save("minnorm_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* mlpack::RandomSeed(SEED); */
    /* { */
    /*   src::LCurve<LinearRegression<pinv>,mlpack::MSE> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,Ds[i]*stddev*stddev/arma::norm(arma::ones<arma::Row<DTYPE>>(1,Ds[i]),2),false); */
    /*   lc.test_errors_.save("reg_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */

    /* mlpack::RandomSeed(SEED); */
    /* { */
    /*   src::LCurve<EigVal<>,utils::DummyReg> lc(Ns,rep,true,false,true); */
    /*   lc.Bootstrap(dataset.inputs_,dataset.labels_,stddev); */
    /*   lc.test_errors_.save("eigen_rank"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */
  }
  PRINT_TIME(timer.toc());

  return 0;
}
