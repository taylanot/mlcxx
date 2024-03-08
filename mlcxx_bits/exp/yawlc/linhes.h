/**
 * @file linhes.h
 * @author Ozgur Taylan Turan
 *
 * Linear model expected Hessian eigenvalue calculation
 *
 */

#ifndef YAWLC_LINHES_H 
#define YAWLC_LINHES_H

namespace experiments {
namespace yawlc {

void linhes ( )
{
  utils::data::regression::Dataset dataset(conf::D, conf::N);

  dataset.Generate(1.,0.,conf::type);
  if (conf::clip_y)
    dataset.labels_.clamp(0., arma::datum::inf);

  if (conf::clip_x)
    dataset.inputs_.clamp(0., arma::datum::inf);


  arma::mat H_min(conf::replinhes,conf::Ns.n_elem);
  arma::mat Ht_min(conf::replinhes,conf::Ns.n_elem);

  arma::mat H_max(conf::replinhes,conf::Ns.n_elem);
  arma::mat Ht_max(conf::replinhes,conf::Ns.n_elem);

  PRINT("RUNNNING...");

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(conf::Ns.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(conf::replinhes); j++)
    {
      const auto res = utils::data::Split(dataset.inputs_,
                                          dataset.labels_,
                                          conf::Ns(i));
      arma::mat Xtrn = std::get<0>(res); 
      arma::mat X = arma::join_vert(Xtrn, arma::ones<arma::rowvec>(Xtrn.n_cols));
      arma::mat Xtst = std::get<1>(res); 
      arma::mat Xt = arma::join_vert(Xtst, arma::ones<arma::rowvec>(Xtst.n_cols));

      arma::mat eigval_trn = arma::eig_sym(arma::pinv((2 * X*X.t() / conf::Ns(i))));
      arma::mat eigval_tst = arma::eig_sym(arma::pinv((2 * Xt*Xt.t() / Xt.n_cols)));

      H_min(j,i) = eigval_trn(0);
      H_max(j,i) = eigval_trn(1);
      Ht_min(j,i) = eigval_tst(0);
      Ht_max(j,i) = eigval_tst(1);
    }
  }
  arma::mat Hm_min = arma::mean(H_min,0);
  arma::mat Hm_max = arma::mean(H_max,0);

  arma::mat Hs_min = arma::stddev(H_min,0);
  arma::mat Hs_max = arma::stddev(H_max,0);

  arma::mat Hm = arma::join_vert(Hm_min, Hm_max);
  arma::mat Hs = arma::join_vert(Hs_min, Hs_max);

  arma::mat Htm_min = arma::mean(Ht_min,0);
  arma::mat Htm_max = arma::mean(Ht_max,0);

  arma::mat Hts_min = arma::stddev(Ht_min,0);
  arma::mat Hts_max = arma::stddev(Ht_max,0);

  arma::mat Htm = arma::join_vert(Htm_min, Htm_max);
  arma::mat Hts = arma::join_vert(Hts_min, Hts_max);

  utils::Save("hess.csv",Hm,Hs); 
  utils::Save("hess_act.csv",Htm,Hts); 

  PRINT_VAR(H_min);
  PRINT_VAR(Ht_min);
}


void linearmodel ( )
{
  utils::data::regression::Dataset dataset(conf::D, conf::N);

  dataset.Generate(1.,0.,conf::type);

  if (conf::clip_y)
    dataset.labels_.clamp(0., arma::datum::inf);

  if (conf::clip_x)
    dataset.inputs_.clamp(0., arma::datum::inf);

  arma::mat error(conf::replinhes,conf::Ns.n_elem);

  arma::mat w1(conf::replinhes,conf::Ns.n_elem);
  arma::mat w2(conf::replinhes,conf::Ns.n_elem);

  PRINT("RUNNNING-linearmodel...");

  std::filesystem::path root = "yawlc-experiments";
  std::filesystem::path subdirdata = "data";
  std::filesystem::path subdiropt = "optim";

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(conf::Ns.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(conf::replinhes); j++)
    {
      const auto data = utils::data::Split(dataset.inputs_,
                                          dataset.labels_,
                                          conf::Ns(i));
      arma::mat X1 = std::get<0>(data); 
      arma::mat Ytrn = std::get<2>(data); 
      arma::mat Xtrn = arma::join_vert(X1, arma::ones<arma::rowvec>(X1.n_cols));

      std::string filename = std::to_string(j)+".csv";

      std::filesystem::create_directories(root/subdirdata/std::to_string(i));
      utils::Save(root/subdirdata/std::to_string(i)/filename, X1,Ytrn);

      arma::mat X2 = std::get<1>(data); 
      arma::mat Ytst = std::get<3>(data); 
      arma::mat Xtst = arma::join_vert(X2, arma::ones<arma::rowvec>(X2.n_cols));

      arma::mat W = arma::pinv(Xtrn*Xtrn.t())*Xtrn*Ytrn.t();
      arma::rowvec res = (W.t()*Xtst-Ytst);
      //PRINT_VAR(res);

      error (j,i) = arma::dot(res,res) / double(Xtst.n_cols) ;

      w1(j,i) =  W(0,0);
      w2(j,i) =  W(1,0);
    

    }
  }

  std::filesystem::create_directories(root/subdiropt);
  utils::Save(root/"error.csv",error);
  utils::Save(root/subdiropt/"w1.csv",w1);
  utils::Save(root/subdiropt/"w2.csv",w2);
}

void linheslc ( )
{
  utils::data::regression::Dataset dataset(conf::D, conf::N);

  dataset.Generate(1.,0.,conf::type);
  if (conf::clip_y)
    dataset.labels_.clamp(0., arma::datum::inf);

  if (conf::clip_x)
    dataset.inputs_.clamp(0., arma::datum::inf);


  src::regression::LCurve<mlpack::LinearRegression,
                          mlpack::MSE> lcurve(conf::Ns,conf::replearncurve);

  std::filesystem::path filename = "lc.csv"; 

  lcurve.Generate(filename, dataset.inputs_,
                  arma::conv_to<arma::rowvec>::from(dataset.labels_));
}

void stats ( )
{
  utils::data::regression::Dataset dataset(conf::D, conf::N);

  dataset.Generate(1.,0.,conf::type);
  if (conf::clip_y)
    dataset.labels_.clamp(0., arma::datum::inf);

  if (conf::clip_x)
    dataset.inputs_.clamp(0., arma::datum::inf);


  PRINT("RUNNNING yawl::stats ...");


  arma::mat cov_samp(conf::replinhes, conf::Ns.n_elem);
  arma::mat cov_unsamp(conf::replinhes, conf::Ns.n_elem);

  arma::mat mean_samp(conf::replinhes, conf::Ns.n_elem);
  arma::mat mean_unsamp(conf::replinhes, conf::Ns.n_elem);

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(conf::Ns.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(conf::replinhes); j++)
    {
      const auto res = utils::data::Split(dataset.inputs_,
                                          dataset.labels_,
                                          conf::Ns(i));
      arma::mat Xtrn = std::get<0>(res); 
      arma::mat X = arma::join_vert(Xtrn, arma::ones<arma::rowvec>(Xtrn.n_cols));
      arma::mat Xtst = std::get<1>(res); 
      arma::mat Xt = arma::join_vert(Xtst, arma::ones<arma::rowvec>(Xtst.n_cols));

      arma::mat cov1 = arma::cov(X.t());
      arma::mat cov2 = arma::cov(Xt.t());


      //arma::mat cov1 = arma::cov(Xtrn.t());
      //arma::mat cov2 = arma::cov(Xtst.t());

      //arma::mat mean1 = arma::mean(Xtrn.t());
      //arma::mat mean2 = arma::mean(Xtst.t());

      cov_samp(j,i) = arma::eig_sym(cov1)(1);
      cov_unsamp(j,i) = arma::eig_sym(cov2)(1);

      //mean_samp(j,i) = mean1(0,0);
      //mean_unsamp(j,i) = mean2(0,0);

      //PRINT_VAR(cov1);
      //PRINT_VAR(cov2);
    }
  }

  utils::Save("eig_cov_samp.csv",cov_samp); 
  utils::Save("eig_cov_unsamp.csv",cov_unsamp); 

  //utils::Save("eig_mean_samp.csv",mean_samp); 
  //utils::Save("eig_mean_unsamp.csv",mean_unsamp); 
}
} // ywalc namespace
} // experiments

#endif
