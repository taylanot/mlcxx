/**
 * @file new_spkr_helper.cpp
 * @author Ozgur Taylan Turan
 *
 */

#include <headers.h>

using namespace experiments::llc;

class PCAREAD
{
  public:
  
  PCAREAD ( const size_t& M=0 ) : M_(0)
  {
    arma::mat data;
    data.load(conf::train_path/conf::train_file);   
    arma::inplace_trans(data);
    X_ = data.row(0);
    mean_ = data.row(1);
    M_ = data.n_rows-2;
    y_ = data.rows(2,data.n_rows-1);
  }
  
  arma::mat Predict ( const arma::mat& X ) 
  {
    arma::uvec idx, tempid;
    idx.resize(X.n_cols);
    for ( size_t i=0; i < X.n_cols; i++)
    {
      tempid = arma::find(X_ == double(X(0,i)));
      idx(i) = tempid(0);
    }
    return y_.cols(idx);
  }

  size_t GetM ( ) 
  {
    return M_;
  }

  void  SetM ( size_t M ) 
  {
    M_ = M;
  }

  arma::rowvec Mean ( const arma::mat& X )
  {
    arma::mat labels;
    arma::rowvec temp;

    arma::uvec idx, tempid;
    idx.resize(X.n_cols);
    for ( size_t i=0; i< X.n_cols; i++)
    {
      tempid = arma::find(X_ == double(X(0,i)));
      idx(i) = tempid(0);
    }

    return mean_.cols(idx);
  }

  private:
  arma::mat X_;
  arma::mat y_;
  arma::rowvec mean_;
  size_t M_;
};

void extract_pca(const std::vector<std::string>& names)
{
  arma::mat train_data;
  arma::field<std::string> train_header;
  {
    using namespace experiments::llc;
    train_data.load(arma::csv_name(conf::train_path/"train.csv",
          train_header));
  }

  arma::uvec idx;
  if (names.size() == 4)
  {
    idx = utils::select_headers(train_header,{names[0],names[2]});
  }
  else if (names.size() == 2)
  {
    idx = utils::select_header(train_header,names[0]);
  }
  arma::mat yt = train_data.cols(idx); arma::inplace_trans(yt);
  arma::mat xt = train_data.col(0); arma::inplace_trans(xt);

  arma::mat ysmooth = algo::functional::kernelsmoothing
    <mlpack::GaussianKernel>(xt,yt,1);
  arma::mat mean = arma::mean(yt,0);
  arma::mat funcs;
  auto res = algo::functional::ufpca(xt,ysmooth,0.95);
  arma::mat eigfuncs = std::get<1>(res);
  arma::mat pca = arma::join_horiz(xt.t(),mean.t(),eigfuncs.t());
  pca.save(arma::csv_name(experiments::llc::conf::train_path/"pca_.csv"));

}

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{

  /* arma::mat data; */
  /* arma::field<std::string> header; */
  /* std::filesystem::path dir; */
  /* { */
  /*   using namespace experiments::llc; */
  /*   dir = conf::train_root/"classification"; */
  /*   /1* dir = conf::train_root/"regression"; *1/ */
  /*   data.load(arma::csv_name(dir/"train.csv",header)); */   
  /* } */

  /* arma::mat x = data.col(0);arma::inplace_trans(x); */
  /* arma::mat y = data.cols(1,data.n_cols-1);arma::inplace_trans(y); */
  /* arma::mat ysmooth = algo::functional::kernelsmoothing */
  /*   <mlpack::GaussianKernel>(x,y,1); */
  /* arma::mat mean = arma::mean(y,0); */
  /* arma::mat funcs; */
  /* auto res = algo::functional::ufpca(x,ysmooth,0.99); */
  /* arma::mat eigfuncs = std::get<1>(res); */
  /* arma::mat pca = arma::join_horiz(x.t(),mean.t(),eigfuncs.t()); */
  /* pca.save(arma::csv_name(dir/"pca_smooth.csv")); */


  /* std::vector<std::string> sub_groups = {"gausskernelridge", */
  /*   "laplacekernelridge","saw","dgp","nn_adam","linear"}; */
  /* /1* std::vector<std::string> sub_groups = {"nmc","nnc","qdc","ldc"}; *1/ */

  /* arma::mat data; */
  /* arma::field<std::string> header; */
  /* std::filesystem::path dir; */
  /* { */
  /*   using namespace experiments::llc; */
  /*   dir = conf::train_root/"regression"; */
  /*   /1* dir = conf::train_root/"classification"; *1/ */
  /*   data.load(arma::csv_name(dir/"train.csv",header)); */   
  /* } */
  /* arma::mat x = data.col(0);arma::inplace_trans(x); */
  /* arma::mat y = data.cols(1,data.n_cols-1);arma::inplace_trans(y); */

  /* arma::mat mean = arma::mean(y,0); */
  /* arma::mat funcs; */
  /* size_t counter = 0; */
  /* for (std::string name : sub_groups) */
  /* { */
  /*   arma::uvec ids = utils::select_header(header,name); */
  /*   arma::mat ys = data.cols(ids);arma::inplace_trans(ys); */
  /*   arma::mat ysmooth = algo::functional::kernelsmoothing */
  /*   <mlpack::GaussianKernel>(x,ys,1); */
  /*   auto res = algo::functional::ufpca(x,ysmooth,0.99); */
  /*   arma::mat eigfuncs = std::get<1>(res); */
  /*   if (counter == 0) */
  /*     funcs = eigfuncs; */
  /*   else */
  /*     funcs = arma::join_vert(funcs,eigfuncs); */
  /*   counter++; */
  /* } */
  /* arma::mat pca = arma::join_horiz(x.t(),mean.t(),funcs.t()); */
  /* pca.save(arma::csv_name(dir/"pca_model_smooth.csv")); */

  //
  //std::vector<std::string> sub_groups = {"gausskernelridge","laplacekernelridge",
  //                       "nn_adam","linear"};
  //std::vector<std::string> sub_sub_groups = {"elin","esine","esinc"};
  //std::vector<std::string> sub_groups = {"qdc","ldc","nnc","nmc"};
  //std::vector<std::string> sub_sub_groups = {"banana","ddip","rdip","gaussian"};
  //std::vector<std::string> sub_groups = {"ldc"};
  //std::vector<std::string> sub_sub_groups = {"banana"};

  //for (std::string name : sub_groups)
  //{
  //  arma::field<std::string> header;
  //  arma::mat data;
  //  {
  //    using namespace experiments::llc;
  //    data.load(arma::csv_name(conf::train_root/"classification"/conf::train_file,
  //                                                                      header));   
  //  }
  ////arma::uvec ids = utils::select_headers(header,{"saw","linear"});
  //  for (std::string name2 : sub_sub_groups)
  //  {
  //    arma::uvec ids = utils::select_headers(header,{name,name2});
  //    PRINT_VAR(arma::size(ids));
  //    arma::mat ys = data.cols(ids);arma::inplace_trans(ys);
  //    arma::mat x = data.col(0);arma::inplace_trans(x);

  //    auto res = algo::functional::ufpca(x,ys,0.99);
  //    arma::mat eigfuncs = std::get<1>(res);
  //    PRINT(arma::size(eigfuncs));
  //  }
  //}
  // determine and save the subsets first N, mean, pca1,M


  //PRINT(arma::conv_to<arma::Col<std::string>>::from(header))
  //PRINT(header.t());
  //arma::uvec ids = select(header,"gausskernelridge");
  //PRINT(ids);
  //for (int i = 1; i<ids.n_elem; i++)
  //{ 
  //  PRINT(header(size_t(ids[ids.n_elem-i])));
  //}
  
  /* std::filesystem::path spkr_path = "SPKR2"; */
  /* arma::field<std::string> header; */
  /* arma::mat data; */
  /* data.load(arma::csv_name(conf::test_path/conf::test_file, header)); */   
  /* arma::inplace_trans(data); */
  /* arma::mat inputs = data.row(0); */
  /* arma::mat labels = data.rows(1,data.n_rows-1); */
  /* arma::mat X_train, X_test; arma::rowvec y_train, y_test; */


  /* arma::mat X_test_at(1,1); */
  /* arma::rowvec y_test_at(1); */
  /* bool check; */
  /* //for ( size_t i=0; i<conf::Ntrns.n_cols; i++ ) */
  /* for ( size_t i=0; i<1; i++ ) */
  /* { */
  /*   arma::rowvec label; */
  /*   std::filesystem::path write_path = */ 
  /*                                  std::to_string(size_t(conf::Ntrns(i))); */
    
  /*   std::filesystem::path path_clean = conf::exp_id/spkr_path/write_path/ */
  /*                   utils::remove_path(conf::test_path, conf::test_root); */


  /*   check = std::filesystem::exists(path_clean/conf::err_file); */
  /*   arma::mat error_at; */
  /*   arma::mat error; */
  /*   arma::mat predictions(labels.n_rows,conf::Ns.n_elem); */

  /*   if ( !check ) */
  /*   { */
  /*     std::filesystem::create_directories(path_clean); */
  /*     for ( size_t j=0; j<labels.n_rows; j++) */
  /*     //for ( size_t j=0; j<1; j++) */
  /*     { */
  /*       PRINT(j) */
  /*       label = labels.row(j); */
  /*       error_at.resize(1,labels.n_rows); */
  /*       error.resize(1,labels.n_rows); */

  /*       X_test = inputs.cols(conf::Ntrns(i),inputs.n_cols-1); */
  /*       y_test = label.cols(conf::Ntrns(i),label.n_cols-1); */

  /*       X_train = inputs.cols(0,conf::Ntrns(i)); */
  /*       y_train = label.cols(0,conf::Ntrns(i)); */

  /*       mlpack::hpt::HyperParameterTuner */
  /*        <algo::regression::SemiParamKernelRidge */
  /*                 <mlpack::GaussianKernel,PCAREAD>, */
  /*         mlpack::cv::MSE, */
  /*         mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train); */
  /*       double best_lambda, best_l; */
  /*       size_t best_npc = 0; */
        
  /*       std::tie(best_lambda, best_l) = */ 
  /*                         hpt.Optimize(conf::lambdas,mlpack::Fixed(best_npc),conf::ls); */

  /*       algo::regression::SemiParamKernelRidge<mlpack::GaussianKernel, */
  /*                                              PCAREAD> */ 
  /*                 model(X_train, y_train,best_lambda, best_npc, best_l); */

  /*       X_test_at(0,0) = inputs(conf::idx); */
  /*       y_test_at(0) = label(conf::idx); */

  /*       error_at(j) = model.ComputeError(X_test_at,y_test_at); */
  /*       error(j) = model.ComputeError(inputs,label); */
  /*       arma::rowvec temp_y; */
  /*       model.Predict(inputs,temp_y); */
  /*       predictions.row(j) = temp_y; */ 
  /*     } */
  /*       arma::inplace_trans(predictions); */
  /*       predictions = arma::join_rows( */
  /*         arma::conv_to<arma::mat>::from(conf::Ns).t(),predictions); */

  /*       error_at.save(arma::csv_name(path_clean/conf::at_file, */
  /*                                  header.cols(1,labels.n_rows))); */
  /*       error.save(arma::csv_name(path_clean/conf::err_file, */
  /*                                  header.cols(1,labels.n_rows))); */
  /*       predictions.save(arma::csv_name(path_clean/conf::pred_file, */
  /*                                  header.cols(0,labels.n_rows))); */
  /*   } */
  /* } */

  std::filesystem::path spkr_path = "SPKR2_gsmooth95log";
  arma::field<std::string> header;
  arma::mat data;
  data.load(arma::csv_name(conf::test_path/conf::test_file, header));   
  arma::inplace_trans(data);
  arma::mat inputs = data.row(0);
  arma::mat labels = data.rows(1,data.n_rows-1);
  arma::mat X_train, X_test; arma::rowvec y_train, y_test;


  arma::mat X_test_at(1,1);
  arma::rowvec y_test_at(1);
  bool check;
  for ( size_t i=0; i<conf::Ntrns.n_cols; i++ )
  {
    arma::rowvec label;
    std::filesystem::path write_path = 
                                   std::to_string(size_t(conf::Ntrns(i)));
    
    std::filesystem::path path_clean = conf::exp_id/spkr_path/write_path/
                    utils::remove_path(conf::test_path, conf::test_root);


    check = std::filesystem::exists(path_clean/conf::err_file);
    arma::mat error_at;
    arma::mat error;
    arma::mat predictions(labels.n_rows,conf::Ns.n_elem);

    if ( !check )
    {
      std::filesystem::create_directories(path_clean);
      for ( size_t j=0; j<labels.n_rows; j++)
      //for ( size_t j=0; j<1; j++)
      {
        std::vector<std::string> names = utils::split_path(header(j+1));
        extract_pca(names);

        label = labels.row(j);
        error_at.resize(1,labels.n_rows);
        error.resize(1,labels.n_rows);

        X_test = inputs.cols(conf::Ntrns(i),inputs.n_cols-1);
        y_test = label.cols(conf::Ntrns(i),label.n_cols-1);

        X_train = inputs.cols(0,conf::Ntrns(i));
        y_train = label.cols(0,conf::Ntrns(i));

        mlpack::hpt::HyperParameterTuner
         <algo::regression::SemiParamKernelRidge2
                  <mlpack::GaussianKernel,PCAREAD>,
          mlpack::cv::MSE,
          mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train);
        double best_lambda, best_l;
        size_t best_npc = 0;
        
        std::tie(best_lambda, best_l) = 
                          hpt.Optimize(conf::lambdas,mlpack::Fixed(best_npc),conf::ls);

        algo::regression::SemiParamKernelRidge2<mlpack::GaussianKernel,
                                               PCAREAD> 
                  model(X_train, y_train,best_lambda, best_npc, best_l);

        X_test_at(0,0) = inputs(conf::idx);
        y_test_at(0) = label(conf::idx);

        error_at(j) = model.ComputeError(X_test_at,y_test_at);
        error(j) = model.ComputeError(inputs,label);
        arma::rowvec temp_y;
        model.Predict(inputs,temp_y);
        predictions.row(j) = temp_y; 
      }
        arma::inplace_trans(predictions);
        predictions = arma::join_rows(
          arma::conv_to<arma::mat>::from(conf::Ns).t(),predictions);

        error_at.save(arma::csv_name(path_clean/conf::at_file,
                                   header.cols(1,labels.n_rows)));
        error.save(arma::csv_name(path_clean/conf::err_file,
                                   header.cols(1,labels.n_rows)));
        predictions.save(arma::csv_name(path_clean/conf::pred_file,
                                   header.cols(0,labels.n_rows)));
    }
  }
   
  return 0; 
}
