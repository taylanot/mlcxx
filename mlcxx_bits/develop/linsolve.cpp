/**
 * @file linsolve.cpp
 * @author Ozgur Taylan Turan
 *
 * Looking at other linear solvers mostly iterative compared to minimum norm 
 * and least squares solutions.
 *
 */
#define DTYPE double
#include <headers.h>

struct gmres
{
  template<class T=DTYPE>
  arma::Col<T> operator()(const arma::Mat<T>& A, const arma::Row<T>& b,
                          T tol = 1e-6, size_t max_iter = 10000) 
  {
    int n = A.n_rows;
    size_t m = A.n_cols;
    arma::Col<T> x(n);
    arma::Col<T> r =(b - x.t()*A).t() ;
    T beta = arma::norm(r,2);
    
    if (beta < tol) return x;
    
    arma::Mat<T> V(n, m + 1);
    V.col(0) = r / beta;
    
    arma::Mat<T> H (m + 1, m);
    arma::Col<T> cs(m);
    arma::Col<T> sn(m);
    arma::Col<T> e1(m + 1);
    e1(0) = beta;
    
    for (size_t j = 0; j < std::min(m, max_iter); ++j)
    {
      arma::Col<T> w = A * V.col(j);
      
      for (size_t i = 0; i <= j; ++i)
      {
        H(i, j) = arma::dot(V.col(i), w);
        w -= H(i, j) * V.col(i);
      }
      
      H(j + 1, j) = arma::norm(w);

      if (H(j + 1, j) < tol) break;
      
      V.col(j + 1) = w / H(j + 1, j);
      
      // Apply Givens rotation
      for (size_t i = 0; i < j; ++i)
      {
        T temp = cs(i) * H(i, j) + sn(i) * H(i + 1, j);
        H(i + 1, j) = -sn(i) * H(i, j) + cs(i) * H(i + 1, j);
        H(i, j) = temp;
      }
      
      T r = arma::norm(H(arma::span(j, j + 1), j));
      cs(j) = H(j, j) / r;
      sn(j) = H(j + 1, j) / r;
      H(j, j) = r;
      H(j + 1, j) = 0.0;
      e1(j + 1) = -sn(j) * e1(j);
      e1(j) = cs(j) * e1(j);
      
      // Check for convergence
      if (abs(e1(j + 1)) < tol)
      {
        arma::Col<T> y = arma::solve(H(arma::span(0, j), arma::span(0, j)), e1(arma::span(0, j)));
        x += V.cols(0, j) * y;
        return x;
      }
    }
    
    // Solve for the final x
    arma::Col<T> y = solve(H(arma::span(0, m - 1), arma::span(0, m - 1)), e1(arma::span(0, m - 1)));
    x += V.cols(0, m - 1) * y;
    return x;
  }
};

struct  conjgrad
{
  template<class T=DTYPE>
  arma::Col<T> operator()(const arma::Mat<T>& A, const arma::Row<T>& b,
                          T tol = 1e-6, size_t max_iter = 1000) 
  {
    arma::Row<T> x0(arma::size(b),arma::fill::ones);
    arma::Row<T> r = (b - x0*A) ;
    arma::Row<T> p = r;
    T rsold = arma::dot(r, r);
    for (size_t i = 0; i < max_iter; ++i)
    {
      arma::Row<T> Ap = p * A ;
      T alpha = rsold / arma::dot(p, Ap);
      x0 = x0 + alpha * p;
      r = r - alpha * Ap;
      T rsnew = arma::dot(r, r);
      if (sqrt(rsnew) < tol)
          break;

      p = r + (rsnew / rsold) * p;
      rsold = rsnew;
    }
    return x0.t();
  }
};

struct cdescent
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b,
                        size_t maxproj= 100000, T tol=1e-6)
  {
    arma::Mat<T> x(arma::size(b),arma::fill::ones);
    size_t i;  
    size_t iter = 0;
    arma::Row<T> dx;
    while (iter++ < maxproj)
    {
      arma::Row<T> e(arma::size(b));
      i = arma::randi(arma::distr_param(0,b.n_elem-1)); 
      e[i] = 1.;
      dx = (b[i]-(arma::dot(A.col(i),x))) * e /arma::trace(A);
      x += dx;
      if (arma::norm(A*x.t()-b.t(),2) < tol)
        break;
    }
    return x.t();
  }

};

struct cdescentlq
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b,
                        size_t maxproj= 100000, T tol=1e-6)
  {
    arma::Mat<T> x(arma::size(b),arma::fill::ones);
    size_t i;  
    size_t iter = 0;
    arma::Row<T> dx;
    arma::Row<T> e(arma::size(b),arma::fill::ones);
    while (iter++ < maxproj)
    {
      arma::Row<T> e(arma::size(b));
      i = arma::randi(arma::distr_param(0,b.n_elem-1)); 
      e[i] = 1.;
      dx =  arma::dot(A.col(i).t(),(x*A.t()-b)) * e /std::pow(arma::norm(A.col(i),2),2);
      x += dx;
      if (arma::norm(A*x.t()-b.t(),2) < tol)
        break;
    }
    return x.t();
  }
};

struct kaczmarz
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b,
                        size_t maxproj= 10000000, T tol=1e-6)
  {
    arma::Mat<T> x(arma::size(b),arma::fill::ones);
    size_t i;  
    size_t iter = 0;
    arma::Row<T> dx;
    while (iter++ < maxproj)
    {
      i = arma::randi(arma::distr_param(0,b.n_elem-1)); 
      dx = (b[i]-(arma::dot(A.col(i),x))) * A.col(i).t()/std::pow(arma::norm(A.col(i),2),2);
      x += dx;
      
      if (arma::norm(A*x.t()-b.t(),2) < tol)
        break;
    }
    return x.t();
  }
};

struct pinv
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    /* return (b*arma::pinv(A,1)).t() ; */
    /* return arma::solve(A,b.t(),arma::solve_opts::allow_ugly); */
    return arma::solve(A,b.t());
  }
};

struct basispurs 
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    if ( arma::rank(A) < A.n_rows )
    {
      arma::Row<T> x;
      arma::Row<T> c(2*A.n_rows,arma::fill::ones);
      arma::Mat<T> A_ = arma::join_vert(arma::join_horiz(A,-A),arma::join_horiz(-A,A));
      arma::Row<T> b_ = arma::join_horiz(b,-b);

      arma::Mat<T> G;
      arma::Row<T> h;

      opt::linprog(x,c,G,h,A_,b_,true,false);
      return (x.cols(0,(x.n_elem/2)-1)-x.cols((x.n_elem/2),x.n_elem-1)).t();
    }
    else
    {
      return arma::solve(A,b.t());
    }

  }
};

struct basispurs2 
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    size_t numcols = b.n_elem;
    arma::Row<T> x;
    arma::Row<T> c(2*numcols,arma::fill::ones);
    c.cols(c.n_elem/2,c.n_elem-1).fill(0.) ;
    arma::Mat<T> eye = arma::eye<arma::Mat<T>>(numcols, numcols);    
    arma::Mat<T> top = arma::join_horiz(eye,-eye);  
    arma::Mat<T> bottom = arma::join_horiz(-eye,-eye); 

    arma::Mat<T> A_ = arma::join_horiz(A,arma::zeros<arma::Mat<T>>(arma::size(A)));
    arma::Row<T> b_ = b;

    arma::Mat<T> G = arma::join_vert(top, bottom); 
    arma::Row<T> h(numcols*2);


    opt::linprog(x,c,G,h,A_,b_,false,false);

    return x.cols(0,(x.n_elem/2)-1).t();
    
  }
};

struct norminf
{
  template<class T=DTYPE>
  arma::Col<T> operator()(const arma::Mat<T> A, const arma::Row<T>b) 
  {
    if ( arma::rank(A) < A.n_rows )
    {
      size_t numcols = b.n_elem;
      arma::Row<T> x;
      arma::Row<T> c(numcols+1,arma::fill::zeros);
      c.col(c.n_elem-1).fill(1.) ;


      arma::Mat<T> A_ = arma::join_horiz(A,arma::zeros<arma::Mat<T>>(numcols,1));

      A_ = arma::join_vert(A_,
          arma::join_horiz(-arma::eye<arma::Mat<T>>(numcols,numcols),-arma::ones<arma::Mat<T>>(numcols,1)));
      A_ = arma::join_vert(A_,
          arma::join_horiz(arma::eye<arma::Mat<T>>(numcols,numcols),-arma::ones<arma::Mat<T>>(numcols,1)));
      
      arma::Row<T> b_ = arma::join_horiz(b,arma::zeros<arma::Mat<T>>(1,2*numcols));


      /* arma::Mat<T> G_ = arma::join_horiz(-arma::eye<arma::Mat<T>>(numcols,numcols),-arma::ones<arma::Mat<T>>(numcols,1)); */
      /* G_ = arma::join_vert(G_,arma::join_horiz(arma::eye<arma::Mat<T>>(numcols,numcols),-arma::ones<arma::Mat<T>>(numcols,1))); */
      /* arma::Row<T> h_(G_.n_rows); */
      arma::Mat<T> G_;
      arma::Row<T> h_;

      opt::linprog(x,c,G_,h_,A_,b_,false,false);

      return x.cols(0,b.n_elem-1).t();
    }
    else
    {
      return arma::solve(A,b.t());
    }

  }
};



struct randomsols
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b, size_t num=1) 
  {
    if ( arma::rank(A) <= A.n_rows )
    {
    arma::Col<T> x = arma::solve(A,b.t());
    arma::Mat<T> null = nullspace(A);
    arma::uword nullrank = null.n_cols;

    auto add = arma::Mat<T>(nullrank, num,arma::fill::ones);

    arma::Col<T> sols = x + 
      /* null * arma::randn<arma::Mat<T>>(nullrank, num,arma::distr_param(20,1)); */
      null * (add*1000);
    return sols;
    }
    else
    {
      return arma::solve(A,b.t());
    }
  }

  template<class T=DTYPE>
  arma::Mat<T> nullspace( const arma::Mat<T>& A,
                          T tol = 1e-5)
  {
    // Perform singular value decomposition
    arma::Mat<T> U, V;
    arma::Col<T> s;
    arma::svd(U, s, V, A);

    // Identify the singular values that are effectively zero
    arma::uvec null_mask = arma::find(s <= tol);

    // Extract the corresponding right singular vectors
    arma::Mat<T> null_space_basis = V.cols(null_mask);

    return null_space_basis;
  }
};

template<class SOLVER=pinv,class T=DTYPE>
class LinearRegression
{
public:

  LinearRegression ( bool bias = false ) : bias_(bias) { }

  template<class... Args>
  LinearRegression ( const arma::Mat<T>& X, const arma::Mat<T>& y, Args&... args)
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

    auto b = arma::conv_to<arma::Row<T>>::from(X_ * y.t()) ;
    parameters_ = solver_((X_*X_.t()).eval(),b);
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
  bool bias_ = false;
};

/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path path = ".09_07_23/solver_comparison"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   size_t D = 10; */
/*   data::regression::Dataset dataset(D,2); */
/*   dataset.Generate("RandomLinear",0.); */

/*   arma::fmat A = dataset.inputs_ * dataset.inputs_.t(); */
/*   auto b = arma::conv_to<arma::frowvec>:: */
/*                       from(dataset.inputs_ * dataset.labels_.t()) ; */
/*   pinv pinv_; */
/*   PRINT_VAR(pinv_(A,b).t()); */
/*   PRINT_VAR(arma::norm(pinv_(A,b),2)); */

/*   basispurs sparse_; */
/*   PRINT_VAR(sparse_(A,b).t()); */
/*   PRINT_VAR(arma::norm(sparse_(A,b),2)); */

/*   randomsols random_; */
/*   PRINT_VAR(random_(A,b).t()); */
/*   PRINT_VAR(arma::norm(random_(A,b),2)); */

/*   gmres gmres_; */
/*   PRINT_VAR(gmres_(A,b).t()); */
/*   PRINT_VAR(arma::norm(gmres_(A,b),2)); */


/*   /1* cdescent cdescend_; *1/ */
/*   /1* PRINT_VAR(cdescent_(A,b).t()); *1/ */
/*   /1* PRINT_VAR(arma::norm(cdescent_(A,b),2)); *1/ */

/*   /1* kaczmarz kaczmarz_; *1/ */
/*   /1* PRINT_VAR(kaczmarz_(A,b).t()); *1/ */
/*   /1* PRINT_VAR(arma::norm(kaczmarz_(A,b),2)); *1/ */


/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   size_t D = 11; */
/*   data::regression::Dataset dataset(D,10); */
/*   dataset.Generate(2.,0,"Linear",1.0); */

/*   arma::fmat A = dataset.inputs_ * dataset.inputs_.t(); */
/*   arma::fmat y = arma::randn<arma::fmat>(D).t() * dataset.inputs_; */

/*   auto b = arma::conv_to<arma::frowvec>:: */
/*                       from(dataset.inputs_ * y.t()) ; */
/*   pinv pinv_; */
/*   PRINT_VAR(pinv_(A,b).t()); */
/*   PRINT_VAR(arma::norm(pinv_(A,b),2)); */

/*   cdescent cdescent_; */
/*   PRINT_VAR(cdescent_(A,b).t()); */
/*   PRINT_VAR(arma::norm(cdescent_(A,b),2)); */

/*   cdescentlq cdescentlq_; */
/*   PRINT_VAR(cdescentlq_(A,b).t()); */
/*   PRINT_VAR(arma::norm(cdescentlq_(A,b),2)); */

/*   kaczmarz kaczmarz_; */
/*   PRINT_VAR(kaczmarz_(A,b).t()); */
/*   PRINT_VAR(arma::norm(kaczmarz_(A,b),2)); */

/*   conjgrad conjgrad_; */
/*   PRINT_VAR(conjgrad_(A,b).t()); */
/*   PRINT_VAR(arma::norm(conjgrad_(A,b),2)); */


/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path path = ".09_07_23/solver_comparison_new_"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   arma::wall_clock timer; */
/*   timer.tic(); */


/*   arma::irowvec Ds = {1,5,10,20}; */
/*   size_t rep = 100; */
/*   for (size_t i=0;i<Ds.n_elem;i++) */
/*   { */
/*     data::regression::Dataset dataset(Ds(i),10000); */
/*     dataset.Generate("Linear",0.5); */

/*     arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,50); */

/*     src::LCurve<LinearRegression<pinv>,mlpack::MSE> lc(Ns,rep,true,false,true); */
/*     src::LCurve<LinearRegression<kaczmarz>,mlpack::MSE> lc_(Ns,rep,true,false,true); */
/*     /1* src::LCurve<LinearRegression<cdescent>,mlpack::MSE> lc__(Ns,rep,true,false,true); *1/ */
/*     /1* src::LCurve<LinearRegression<conjgrad>,mlpack::MSE> lc___(Ns,rep,true,false,true); *1/ */
/*     src::LCurve<LinearRegression<basispurs>,mlpack::MSE> lc____(Ns,rep,true,false,true); */
/*     src::LCurve<LinearRegression<randomsols>,mlpack::MSE> lc_____(Ns,rep,true,false,true); */
/*     lc.Bootstrap(dataset.inputs_,dataset.labels_,true); */
/*     lc_.Bootstrap(dataset.inputs_,dataset.labels_,true); */
/*     /1* lc__.Bootstrap(dataset.inputs_,dataset.labels_,true); *1/ */
/*     /1* lc___.Bootstrap(dataset.inputs_,dataset.labels_,true); *1/ */
/*     lc____.Bootstrap(dataset.inputs_,dataset.labels_,true); */
/*     lc_____.Bootstrap(dataset.inputs_,dataset.labels_,true); */
/*     arma::fmat error; */
/*     error = lc.test_errors_; */
/*     error.save("pinv_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
/*     error = lc_.test_errors_; */
/*     error.save("kaczmarz_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
/*     /1* error = lc__.test_errors_; *1/ */
/*     /1* error.save("cdescent_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); *1/ */
/*     /1* error = lc___.test_errors_; *1/ */
/*     /1* error.save("conjgrad_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); *1/ */
/*     error = lc____.test_errors_; */
/*     error.save("basispurs_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
/*     error = lc_____.test_errors_; */
/*     error.save("random_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */

/*   } */
/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path path = ".09_07_23/solver_comparison_random_fixedtol"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   arma::wall_clock timer; */
/*   timer.tic(); */


/*   arma::irowvec Ds = {1,5,10,20}; */
/*   size_t rep = 30; */

/*   for (size_t i=0;i<Ds.n_elem;i++) */
/*   { */
/*     data::regression::Dataset dataset(Ds(i),10000); */
/*     dataset.Generate("Linear",0.5); */

/*     arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,30); */

/*     src::LCurve<LinearRegression<pinv>,mlpack::MSE> lc(Ns,rep,true,false,true); */
/*     src::LCurve<LinearRegression<kaczmarz>,mlpack::MSE> lc_(Ns,rep,true,false,true); */
/*     src::LCurve<LinearRegression<basispurs>,mlpack::MSE> lc____(Ns,rep,true,false,true); */
/*     src::LCurve<LinearRegression<randomsols>,mlpack::MSE> lc_____(Ns,rep,true,false,true); */
/*     lc.Bootstrap(dataset.inputs_,dataset.labels_,true); */
/*     lc_.Bootstrap(dataset.inputs_,dataset.labels_,true); */
/*     lc____.Bootstrap(dataset.inputs_,dataset.labels_,true); */
/*     lc_____.Bootstrap(dataset.inputs_,dataset.labels_,true); */
/*     arma::fmat error; */
/*     error = lc.test_errors_; */
/*     error.save("pinv_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
/*     error = lc_.test_errors_; */
/*     error.save("kaczmarz_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
/*     error = lc____.test_errors_; */
/*     error.save("basispurs_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
/*     error = lc_____.test_errors_; */
/*     error.save("random_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */

/*   } */
/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */

int main ( int argc, char** argv )
{
  std::filesystem::path path = EXP_PATH/"27_07_23/pinv_biga";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();


  arma::irowvec Ds = {5,10,20};
  /* arma::irowvec Ds = {5}; */


  size_t rep = 5000;

  for (size_t i=0;i<Ds.n_elem;i++)
  {
    data::regression::Dataset dataset(Ds(i),1000);
    dataset.inputs_*1000;
    /* data::regression::Dataset dataset(Ds(i),3); */
    arma::Row<DTYPE> a(Ds(i));
    int n = Ds(i) * 0.6; // 20% of the weights should be one 
    arma::uvec idx = arma::randperm(Ds(i),n);
    a(idx).fill(1000);
    dataset.Generate(a,.2);

    /* dataset.Generate(std::string("RandomLinear"),.2); */

    arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,30);

    /* LinearRegression<pinv> model(dataset.inputs_,dataset.labels_); */
    /* LinearRegression<norminf> model2(dataset.inputs_,dataset.labels_); */

    /* PRINT_VAR(model.Parameters()); */
    /* PRINT_VAR(model2.Parameters()); */

    /* LinearRegression<pinv> model_(dataset.inputs_,dataset.labels_); */
    /* PRINT_VAR(a); */
    /* PRINT_VAR(model_.Parameters()); */

    {
      src::LCurve<LinearRegression<pinv>,mlpack::MSE> lc(Ns,rep,true,false,true);
      lc.Bootstrap(dataset.inputs_,dataset.labels_,true);
      lc.test_errors_.save("pinv_"+std::to_string(Ds(i))+".csv",arma::csv_ascii);
    }

    /* { */
    /*   /1* src::LCurve<LinearRegression<basispurs>,mlpack::MSE> lc____(Ns,rep,true,false,true); *1/ */
    /*   src::LCurve<LinearRegression<norminf>,mlpack::MSE> lc____(Ns,rep,true,false,true); */
    /*   lc____.Bootstrap(dataset.inputs_,dataset.labels_,true); */
    /*   /1* lc____.test_errors_.save("basispurs_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); *1/ */
    /*   lc____.test_errors_.save("norminf_"+std::to_string(Ds(i))+".csv",arma::csv_ascii); */
    /* } */


  }
  PRINT_TIME(timer.toc());

  return 0;
}
