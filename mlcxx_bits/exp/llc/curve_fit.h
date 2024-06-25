/**
 * @file curve_fit.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef LLC_CURVEFIT_H 
#define LLC_CURVEFIT_H

namespace experiments {
namespace llc {

//=============================================================================
// CurveFit : Fits a given objective with given optimizer
//=============================================================================
template<class Objective, class Optimizer> 
class CurveFit2
{
  public:
  CurveFit2 ( size_t n_param, size_t n_train ) : n_restart_(100),
                                                n_param_(n_param),
                                                n_train_(n_train) { };

  CurveFit2 ( size_t n_restart, size_t n_param, size_t n_train ) :
                                                   n_restart_(n_restart),
                                                   n_param_(n_param),
                                                   n_train_(n_train) { };


  std::tuple<arma::vec,double> Fit ( const arma::mat& X,
                                     const arma::rowvec& y )
  {
    arma::mat X_beg = X.cols(0,n_train_-1);
    arma::rowvec y_beg = y.cols(0,n_train_-1);

    arma::mat X_test = X.cols(n_train_,X.n_cols-1);
    arma::rowvec y_test = y.cols(n_train_,y.n_cols-1);

    arma::mat xtrn,xtst;
    arma::rowvec ytrn,ytst;
    mlpack::data::Split ( X_beg, y_beg, xtrn,xtst,ytrn,ytst, conf::valid );

    train_obj_ = Objective(xtrn,ytrn);
    fit_obj_= Objective(X_beg,y_beg);
    opt_.MaxIterations() = max_iter_;
    arma::vec theta(n_param_);
    loss_.resize(n_restart_);

    for ( size_t i=0; i<n_restart_; i++ )
    {
     theta.randn();
     opt_.Optimize(train_obj_, theta);
     param_[i] = theta;
     loss_(i) = train_obj_.ComputeError(theta,xtst,ytst);
    } 
    size_t idx = loss_.index_min(); 
    best_ = param_[idx];

    opt_.Optimize(fit_obj_, best_);

    return std::make_tuple(best_, fit_obj_.Evaluate(best_));

  }

  double ComputeError ( const arma::mat& X,
                        const arma::rowvec& y )
  {
    arma::rowvec res = Predict(X) - y;
    return arma::dot(res,res)/res.n_elem;
  }

  double ComputeErrorAt ( const arma::mat& X,
                          const arma::rowvec& y,
                          const size_t& idx )
  {
    arma::mat X_test = X.col(idx);
    arma::rowvec y_test = y.col(idx);

    return ComputeError(X_test, y_test);
  }
  arma::rowvec Predict ( const arma::mat& Xp )
  {
    return fit_obj_.Predict(best_,Xp);
  }
  
  public:
  size_t n_restart_;
  size_t n_param_;
  size_t max_iter_;
  size_t n_train_;
  arma::vec loss_;
  std::map<int,arma::vec> param_;
  arma::vec best_;
  Optimizer opt_;
  Objective train_obj_;
  Objective fit_obj_;
};
//=============================================================================
// CurveFit : Fits a given objective with given optimizer
//=============================================================================
template<class Objective, class Optimizer> 
class CurveFit
{
  public:
  CurveFit ( size_t n_param, size_t n_train ) : n_restart_(100),
                                                n_param_(n_param),
                                                n_train_(n_train) { };

  CurveFit ( size_t n_restart, size_t n_param, size_t n_train ) :
                                                   n_restart_(n_restart),
                                                   n_param_(n_param),
                                                   n_train_(n_train) { };

  std::tuple<arma::vec,double> Fit ( const arma::mat& X,
                                     const arma::rowvec& y )
  {
    arma::mat xtrn,xtst;
    arma::rowvec ytrn,ytst;
    mlpack::data::Split ( X, y, xtrn,xtst,ytrn,ytst, conf::valid );

    Objective obj(xtrn,ytrn);
    Objective obj2(X,y);
    Optimizer opt;
    opt.MaxIterations() = max_iter_;
    arma::vec theta(n_param_);
    loss_.resize(n_restart_);

    /* for ( size_t i=0; i<n_restart_; i++ ) */
    /* { */
    /*  theta.randn(); */
    /*  opt.Optimize(obj, theta); */
    /*  param_[i] = theta; */
    /*  loss_(i) = obj.ComputeError(theta,xtst,ytst); */
    /* } */ 
    /* size_t idx = loss_.index_min(); */ 

    /* return std::make_tuple(param_[idx], loss_(idx)); */

    for ( size_t i=0; i<n_restart_; i++ )
    {
     theta.randn();
     opt.Optimize(obj, theta);
     param_[i] = theta;
     loss_(i) = obj.ComputeError(theta,xtst,ytst);
    } 
    size_t idx = loss_.index_min(); 
    theta = param_[idx];

    opt.Optimize(obj2, theta);

    return std::make_tuple(theta, obj.Evaluate(theta));

  }

  double ComputeError ( const arma::mat& X,
                        const arma::rowvec& y )
  {
    arma::mat X_train = X.cols(0,n_train_-1);
    arma::rowvec y_train = y.cols(0,n_train_-1);

    arma::mat X_test = X.cols(n_train_,X.n_cols-1);
    arma::rowvec y_test = y.cols(n_train_,y.n_cols-1);

    auto res = Fit(X_train,y_train);
    arma::vec theta = std::get<0>(res);

    Objective obj(X, y);

    return obj.Evaluate(theta);
  }

  arma::rowvec Predict ( const arma::mat& X,
                         const arma::rowvec& y, 
                         const arma::mat& Xp )
  {
    arma::mat X_train = X.cols(0,n_train_-1);
    arma::rowvec y_train = y.cols(0,n_train_-1);

    auto res = Fit(X_train,y_train);
    arma::vec theta = std::get<0>(res);

    Objective obj(X, y);
    return obj.Predict(theta,Xp);
  }
  
  double ComputeErrorAt ( const arma::mat& X,
                          const arma::rowvec& y,
                          const size_t idx )
  {
    arma::mat X_train = X.cols(0,n_train_-1);
    arma::rowvec y_train = y.cols(0,n_train_-1);

    arma::mat X_test = X.col(idx);
    arma::rowvec y_test = y.col(idx);

    auto res = Fit(X_train,y_train);
    arma::vec theta = std::get<0>(res);

    Objective obj(X_test, y_test);

    return obj.Evaluate(theta);
  }

  public:
  size_t n_restart_;
  size_t n_param_;
  size_t max_iter_;
  size_t n_train_;
  arma::vec loss_;
  std::map<int,arma::vec> param_;
};

//=============================================================================
// WBL4 : Parametric Model Objective 
//=============================================================================
class WBL4
{
  public:

  WBL4 ( ) { } 

  WBL4 ( const arma::mat& X,
         const arma::rowvec& y ) : X_(X), y_(y) { }
  
  double Evaluate (const arma::mat& theta)
  {
    const arma::rowvec res = Residual(theta);
    return arma::dot(res,res)/res.n_cols;
  }
  void Gradient ( const arma::mat& theta,
                  arma::mat& gradient )
  {
    const arma::rowvec res = Residual(theta);
    const arma::mat dfdp1 = theta(1,0)*arma::pow(X_,theta(3,0))
                          %arma::exp(-theta(0,0)*arma::pow(X_,theta(3,0)));
    const arma::mat dfdp2 = -arma::exp(
                                    -theta(0,0)*arma::pow(X_,theta(3,0)));
    const arma::mat dfdp3(1,X_.n_cols, arma::fill::ones);
    const arma::mat dfdp4 = theta(0,0)*theta(1,0)*arma::pow(X_,theta(3,0))
        %arma::log(X_)%arma::exp(-theta(0,0)*arma::pow(X_,theta(3,0)));

    gradient = arma::join_cols(dfdp1*res.t(),
                               dfdp2*res.t(),
                               dfdp3*res.t(),
                               dfdp4*res.t());
    gradient *= 2;
  }

  arma::rowvec Residual(const arma::mat& theta)
  {
    return (-theta(1,0)*arma::exp(- theta(0,0)*arma::pow(X_,theta(3,0)))+
                                                            theta(2,0)) - y_;
  }

  arma::rowvec Predict(const arma::mat& theta, const arma::mat& X)
  {
    return (-theta(1,0)*arma::exp(- theta(0,0)*arma::pow(X,theta(3,0)))+
                                                            theta(2,0)) ;
  }

  double ComputeError(const arma::mat& theta, const arma::mat& X,
                                                    const arma::rowvec& y)
  {
    arma::rowvec res = Predict(theta,X) - y ;
    return arma::dot(res,res)/res.n_cols;
  }
  private:
  
  arma::mat X_;
  arma::rowvec y_;
};

//=============================================================================
// LastOneFit : Fits a given objective to the last observed point
//=============================================================================
class LastOneFit
{
  public:
  LastOneFit ( size_t n_train ) : n_train_(n_train) { };


  void Fit ( const arma::mat& X, const arma::rowvec& y )
  {
    param_ = y.col(n_train_-1).eval()(0,0);
  }

  double ComputeError ( const arma::mat& X,
                        const arma::rowvec& y )
  {
    arma::rowvec res = Predict(X) - y;
    return arma::dot(res,res)/res.n_elem;
  }

  double ComputeErrorAt ( const arma::mat& X,
                          const arma::rowvec& y,
                          const size_t& idx )
  {
    arma::mat X_test = X.col(idx);
    arma::rowvec y_test = y.col(idx);

    return ComputeError(X_test, y_test);
  }

  arma::rowvec Predict ( const arma::mat& Xp )
  {
    arma::rowvec res = arma::ones<arma::rowvec>(Xp.n_cols);
    return res * param_;
  }
  
  public:

  double param_;
  size_t n_train_;
};

//=============================================================================
// MMF4 : Parametric Model Objective 
//=============================================================================
class MMF4
{
  public:

  MMF4 ( ) { } 
  
  MMF4 ( const arma::mat& X,
         const arma::rowvec& y ) : X_(X), y_(y) { }
  
  double Evaluate (const arma::mat& theta)
  {
    const arma::rowvec res = Residual(theta);
    return arma::dot(res,res)/res.n_cols;
  }

  void Gradient ( const arma::mat& theta,
                  arma::mat& gradient )
  {
    const arma::rowvec res = Residual(theta);
    const arma::mat dfdp1 = theta(1,0) / 
                            (theta(1,0) + arma::pow(X_,theta(3,0)));

    const arma::mat dfdp2 = (theta(0,0)-theta(2,0))*arma::pow(X_,theta(3,0)) 
                      / arma::pow(theta(1,0)+arma::pow(X_,theta(3,0)),2);

    const arma::mat dfdp3 = arma::pow(X_,theta(3,0)) / 
                                    (theta(1,0) + arma::pow(X_,theta(1,0)));

    const arma::mat dfdp4 = (theta(2,0) * (theta(2,0)-theta(0,0)) * 
                            arma::pow(X_,theta(3,0)) % arma::log(X_))/
                     arma::pow(theta(1,0)+arma::pow(X_,theta(3,0)), 2);

    gradient = arma::join_cols(dfdp1*res.t(),
                               dfdp2*res.t(),
                               dfdp3*res.t(),
                               dfdp4*res.t());
    gradient *= 2;
  }

  arma::rowvec Residual(const arma::mat& theta)
  {
    return (theta(0,0)*theta(1,0) + theta(2,0)*arma::pow(X_,theta(3,0))) /
                                   (theta(2,0)+arma::pow(X_,theta(3,0)))- y_;
  }

  arma::rowvec Predict(const arma::mat& theta, const arma::mat& X)
  {
    return (theta(0,0)*theta(1,0) + theta(2,0)*arma::pow(X,theta(3,0))) /
                                   (theta(2,0)+arma::pow(X,theta(3,0)));
  }
  double ComputeError(const arma::mat& theta, const arma::mat& X,
                                                    const arma::rowvec& y)
  {
    arma::rowvec res = Predict(theta,X) - y ;
    return arma::dot(res,res)/res.n_cols;
  }
  private:
  
  arma::mat X_;
  arma::rowvec y_;
};

//-----------------------------------------------------------------------------
// run_lastone_fit : Run Curve Fit 
//-----------------------------------------------------------------------------
void run_lastone_fit ( )
{
  std::filesystem::path last_path = "LAST";

  //arma::mat data = utils::Load(conf::test_path/conf::test_file, true);
  arma::field<std::string> header;
  arma::mat data;
  data.load(arma::csv_name(conf::test_path/conf::test_file, header));   
  arma::inplace_trans(data);

  arma::mat X = data.row(0);
  arma::mat ys = data.rows(1,data.n_rows-1);

  for ( size_t i=0; i<conf::Ntrns.n_cols; i++ )
  {
    std::filesystem::path addition = std::to_string(size_t(conf::Ntrns(i)));

    LastOneFit last_fit(conf::Ntrns(i));

    arma::mat last_error(1,ys.n_rows);
    arma::mat last_error_at(1,ys.n_rows);
    arma::mat last_pred(ys.n_cols,ys.n_rows);

    arma::rowvec y;

    bool check;

    std::filesystem::path last_path_clean = 
      utils::remove_path(conf::exp_id/last_path/addition/conf::test_path,
                                    conf::test_root);

    for ( size_t j=0; j<ys.n_rows; j++)
    {
      y = ys.row(j);
      last_fit.Fit(X,y);
      check = std::filesystem::exists(last_path_clean/conf::err_file);
      if ( !check )
      {
        last_error.col(j) = last_fit.ComputeError(X,y);
        last_error_at.col(j) = last_fit.ComputeErrorAt(X,y,conf::idx);
        last_pred.col(j) = last_fit.Predict(X).t();
      }
    }
    
    std::filesystem::create_directories(last_path_clean);

    last_error.save(arma::csv_name(last_path_clean/conf::err_file,
                            header.cols(1,ys.n_rows)));

    last_error_at.save(arma::csv_name(last_path_clean/conf::at_file,
                               header.cols(1,ys.n_rows)));

    last_pred.save(arma::csv_name(last_path_clean/conf::pred_file,
                            header.cols(1,ys.n_rows)));
  }
}
//-----------------------------------------------------------------------------
// run_curve_fit : Run Curve Fit 
//-----------------------------------------------------------------------------
void run_curve_fit ( )
{
  std::filesystem::path mmf4_path = "MMF4_fairest";
  std::filesystem::path wbl4_path = "WBL4_fairest";

  //arma::mat data = utils::Load(conf::test_path/conf::test_file, true);
  arma::field<std::string> header;
  arma::mat data;
  data.load(arma::csv_name(conf::test_path/conf::test_file, header));   
  arma::inplace_trans(data);

  arma::mat X = data.row(0);
  arma::mat ys = data.rows(1,data.n_rows-1);

  for ( size_t i=0; i<conf::Ntrns.n_cols; i++ )
  {
    std::filesystem::path addition = std::to_string(size_t(conf::Ntrns(i)));

    CurveFit2<WBL4, ens::L_BFGS> wbl4_fit(conf::restart_opt,4,conf::Ntrns(i));
    CurveFit2<MMF4, ens::L_BFGS> mmf4_fit(conf::restart_opt,4,conf::Ntrns(i));

    arma::mat mmf4_error(1,ys.n_rows);
    arma::mat wbl4_error(1,ys.n_rows);
    arma::mat mmf4_error_at(1,ys.n_rows);
    arma::mat wbl4_error_at(1,ys.n_rows);
    arma::mat mmf4_pred(ys.n_cols,ys.n_rows);
    arma::mat wbl4_pred(ys.n_cols,ys.n_rows);



    arma::rowvec y;

    bool check;

    std::filesystem::path mmf4_path_clean = 
      utils::remove_path(conf::exp_id/mmf4_path/addition/conf::test_path,
                                    conf::test_root);
    std::filesystem::path wbl4_path_clean = 
      utils::remove_path(conf::exp_id/wbl4_path/addition/conf::test_path,
                                    conf::test_root);

    for ( size_t j=0; j<ys.n_rows; j++)
    /* for ( size_t j=0; j<1; j++) */
    {
      y = ys.row(j);
      wbl4_fit.Fit(X,y);
      mmf4_fit.Fit(X,y);
      check = std::filesystem::exists(wbl4_path_clean/conf::err_file);
      if ( !check )
      {
        wbl4_error.col(j) = wbl4_fit.ComputeError(X,y);
        wbl4_error_at.col(j) = wbl4_fit.ComputeErrorAt(X,y,conf::idx);
        wbl4_pred.col(j) = wbl4_fit.Predict(X).t();
      }
      check = std::filesystem::exists(mmf4_path_clean/conf::err_file);
      if ( !check )
      {
        mmf4_error.col(j) = mmf4_fit.ComputeError(X,y);
        mmf4_error_at.col(j) = mmf4_fit.ComputeErrorAt(X,y,conf::idx);
        mmf4_pred.col(j) = mmf4_fit.Predict(X).t();
      }
    }
    
    std::filesystem::create_directories(mmf4_path_clean);
    std::filesystem::create_directories(wbl4_path_clean);

    mmf4_error.save(arma::csv_name(mmf4_path_clean/conf::err_file,
                            header.cols(1,ys.n_rows)));
    wbl4_error.save(arma::csv_name(wbl4_path_clean/conf::err_file,
                            header.cols(1,ys.n_rows)));

    mmf4_error_at.save(arma::csv_name(mmf4_path_clean/conf::at_file,
                               header.cols(1,ys.n_rows)));
    wbl4_error_at.save(arma::csv_name(wbl4_path_clean/conf::at_file,
                               header.cols(1,ys.n_rows)));

    wbl4_pred.save(arma::csv_name(wbl4_path_clean/conf::pred_file,
                            header.cols(1,ys.n_rows)));
    mmf4_pred.save(arma::csv_name(mmf4_path_clean/conf::pred_file,
                            header.cols(1,ys.n_rows)));



  }
}

/* //----------------------------------------------------------------------------- */
/* // run_curve_fit : Run Curve Fit */ 
/* //----------------------------------------------------------------------------- */
/* void run_curve_fit ( ) */
/* { */
/*   std::filesystem::path mmf4_path = "MMF4_fairer"; */
/*   std::filesystem::path wbl4_path = "WBL4_fairer"; */

/*   //arma::mat data = utils::Load(conf::test_path/conf::test_file, true); */
/*   arma::field<std::string> header; */
/*   arma::mat data; */
/*   data.load(arma::csv_name(conf::test_path/conf::test_file, header)); */   
/*   arma::inplace_trans(data); */

/*   arma::mat X = data.row(0); */
/*   arma::mat ys = data.rows(1,data.n_rows-1); */

/*   for ( size_t i=0; i<conf::Ntrns.n_cols; i++ ) */
/*   { */
/*     std::filesystem::path addition = std::to_string(size_t(conf::Ntrns(i))); */

/*     CurveFit2<WBL4, ens::L_BFGS> wbl4_fit(conf::restart_opt,4,conf::Ntrns(i)); */
/*     CurveFit2<MMF4, ens::L_BFGS> mmf4_fit(conf::restart_opt,4,conf::Ntrns(i)); */

/*     arma::mat mmf4_error(1,ys.n_rows); */
/*     arma::mat wbl4_error(1,ys.n_rows); */
/*     arma::mat mmf4_error_at(1,ys.n_rows); */
/*     arma::mat wbl4_error_at(1,ys.n_rows); */


/*     arma::rowvec y; */

/*     bool check; */

/*     std::filesystem::path mmf4_path_clean = */ 
/*       utils::remove_path(conf::exp_id/mmf4_path/addition/conf::test_path, */
/*                                     conf::test_root); */
/*     std::filesystem::path wbl4_path_clean = */ 
/*       utils::remove_path(conf::exp_id/wbl4_path/addition/conf::test_path, */
/*                                     conf::test_root); */


/*     for ( size_t j=0; j<ys.n_rows; j++) */
/*     /1* for ( size_t j=0; j<1; j++) *1/ */
/*     { */
/*       y = ys.row(j); */
/*       check = std::filesystem::exists(wbl4_path_clean/conf::err_file); */
/*       if ( !check ) */
/*       { */
/*         wbl4_error.col(j) = wbl4_fit.ComputeError(X,y); */
/*         wbl4_error_at.col(j) = wbl4_fit.ComputeErrorAt(X,y,conf::idx); */
/*       } */
/*       check = std::filesystem::exists(mmf4_path_clean/conf::err_file); */
/*       if ( !check ) */
/*       { */
/*         mmf4_error.col(j) = mmf4_fit.ComputeError(X,y); */
/*         mmf4_error_at.col(j) = mmf4_fit.ComputeErrorAt(X,y,conf::idx); */
/*       } */
/*     } */
    
/*     std::filesystem::create_directories(mmf4_path_clean); */
/*     std::filesystem::create_directories(wbl4_path_clean); */

/*     //utils::Save(mmf4_path_clean/conf::err_file,mmf4_error); */
/*     //utils::Save(wbl4_path_clean/conf::err_file,wbl4_error); */

/*     mmf4_error.save(arma::csv_name(mmf4_path_clean/conf::err_file, */
/*                             header.cols(1,ys.n_rows))); */
/*     wbl4_error.save(arma::csv_name(wbl4_path_clean/conf::err_file, */
/*                             header.cols(1,ys.n_rows))); */

/*     mmf4_error_at.save(arma::csv_name(mmf4_path_clean/conf::at_file, */
/*                                header.cols(1,ys.n_rows))); */
/*     wbl4_error_at.save(arma::csv_name(wbl4_path_clean/conf::at_file, */
/*                                header.cols(1,ys.n_rows))); */


/*   } */
/* } */

//-----------------------------------------------------------------------------
// run_curve_fit_pred: Run Curve Fit for a given curve id
//-----------------------------------------------------------------------------
void run_curve_fit_pred( )
{
  size_t id_saw = 0;
  size_t id_ddip = 41;
  std::filesystem::create_directories("results/ddip");
  std::filesystem::create_directories("results/saw");

  arma::field<std::string> header_class;
  arma::mat data_class;
  data_class.load(arma::csv_name(conf::test_path_class/conf::test_file, header_class));   
  arma::inplace_trans(data_class);
  arma::uvec ids_class = select2(header_class,"ddip");

  arma::mat X_class = data_class.row(0);
  arma::mat ys_class = data_class.rows(ids_class.row(id_ddip));
  
  arma::field<std::string> header_reg;
  arma::mat data_reg;
  data_reg.load(arma::csv_name(conf::test_path_reg/conf::test_file, header_reg));   
  arma::inplace_trans(data_reg);
  arma::uvec ids_reg = select2(header_reg,"saw");

  arma::mat X_reg = data_reg.row(0);
  arma::mat ys_reg = data_reg.rows(ids_reg.row(id_saw));

  for ( size_t i=0; i<conf::Ntrns.n_cols; i++ )
  {

    CurveFit<WBL4, ens::L_BFGS> wbl4_fit(conf::restart_opt,4,conf::Ntrns(i));
    CurveFit<MMF4, ens::L_BFGS> mmf4_fit(conf::restart_opt,4,conf::Ntrns(i));

    arma::rowvec y;
    arma::mat X;
    arma::mat mmf4_class_pred(ys_class.n_rows,X_class.n_cols);
    arma::mat wbl4_class_pred(ys_class.n_rows,X_class.n_cols);

    arma::mat mmf4_reg_pred(ys_class.n_rows,X_class.n_cols);
    arma::mat wbl4_reg_pred(ys_class.n_rows,X_class.n_cols);

    for ( size_t j=0; j<ys_class.n_rows; j++)
    {
      y = ys_class.row(i);
      X = X_class;
      wbl4_class_pred.row(j) = wbl4_fit.Predict(X,y,X);
      mmf4_class_pred.row(j) = mmf4_fit.Predict(X,y,X);
    }
    ys_class.save("results/ddip/truth.csv",arma::csv_ascii);
    mmf4_class_pred.save("results/ddip/mmf4.csv",arma::csv_ascii);
    wbl4_class_pred.save("results/ddip/wbl4.csv",arma::csv_ascii);


    for ( size_t j=0; j<ys_reg.n_rows; j++)
    {
      y = ys_reg.row(i);
      X = X_reg;
      wbl4_reg_pred.row(j) = wbl4_fit.Predict(X,y,X);
      mmf4_reg_pred.row(j) = mmf4_fit.Predict(X,y,X);
    }

    ys_reg.save("results/saw/truth.csv",arma::csv_ascii);
    mmf4_reg_pred.save("results/saw/mmf4.csv",arma::csv_ascii);
    wbl4_reg_pred.save("results/saw/wbl4.csv",arma::csv_ascii);
  }

}

void run_curve_fit_at ( )
{
  std::filesystem::path mmf4_path = "MMF4";
  std::filesystem::path wbl4_path = "WBL4";
 
  arma::field<std::string> header;
  arma::mat data;
  data.load(arma::csv_name(conf::test_path/conf::test_file, header));   
  arma::inplace_trans(data);

  arma::mat X = data.row(0);
  arma::mat ys = data.rows(1,data.n_rows-1);

  for ( size_t i=0; i<conf::Ntrns.n_cols; i++ )
  {
    std::filesystem::path addition = std::to_string(size_t(conf::Ntrns(i)));

    CurveFit<WBL4, ens::L_BFGS> wbl4_fit(conf::restart_opt,4,conf::Ntrns(i));
    CurveFit<MMF4, ens::L_BFGS> mmf4_fit(conf::restart_opt,4,conf::Ntrns(i));

    arma::mat mmf4_error_at(1,ys.n_rows);
    arma::mat wbl4_error_at(1,ys.n_rows);

    arma::rowvec y;

    bool check;


    std::filesystem::path mmf4_path_clean = 
      utils::remove_path(conf::exp_id/mmf4_path/addition/conf::test_path,
                                    conf::test_root);
    std::filesystem::path wbl4_path_clean = 
      utils::remove_path(conf::exp_id/wbl4_path/addition/conf::test_path,
                                    conf::test_root);


    for ( size_t j=0; j<ys.n_rows; j++)
    //for ( size_t j=0; j<1; j++)
    {
      y = ys.row(j);

      check = std::filesystem::exists(wbl4_path_clean/conf::at_file);
      if ( !check )
        wbl4_error_at.col(j) = wbl4_fit.ComputeErrorAt(X,y,conf::idx);

      check = std::filesystem::exists(mmf4_path_clean/conf::at_file);
      if ( !check )
        mmf4_error_at.col(j) = mmf4_fit.ComputeErrorAt(X,y,conf::idx);
    }

    std::filesystem::create_directories(mmf4_path_clean);
    std::filesystem::create_directories(wbl4_path_clean);

    mmf4_error_at.save(arma::csv_name(mmf4_path_clean/conf::at_file,
                               header.cols(1,ys.n_rows)));
    wbl4_error_at.save(arma::csv_name(wbl4_path_clean/conf::at_file,
                               header.cols(1,ys.n_rows)));

  }
}

} //namespace llc
} //namespacee experiments

#endif
