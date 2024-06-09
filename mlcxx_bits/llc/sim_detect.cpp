/**
 * @file sim_detect.cpp
 * @author Ozgur Taylan Turan
 *
 */

#include <headers.h>

using namespace experiments::llc;

class PCAREAD
{
  public:
  
  PCAREAD ( const size_t& M=0 ) : M_(M)
  {
    arma::mat data;
    data.load(conf::train_path/"pca_prox_.csv");   
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


void prox_pca( const arma::rowvec& y, const size_t Ntrn )
{
  arma::mat train_data;
  arma::field<std::string> train_header;
  train_data.load(arma::csv_name(conf::train_path/"train.csv",train_header));
  arma::inplace_trans(train_data);

  arma::mat ys = train_data.cols(1,Ntrn+1); 
  arma::rowvec sim_scores = utils::cos_sim(ys,y);
  arma::uvec idx = arma::sort_index(sim_scores,"descend");

  idx = idx.rows(0,conf::n_closest);

  PRINT(arma::size(train_data));

  arma::mat yt = train_data.rows(idx);
  arma::mat xt = train_data.row(0);

  arma::mat ysmooth = algo::functional::kernelsmoothing
    <mlpack::GaussianKernel>(xt,yt,1);

  arma::mat mean = arma::mean(yt,0);
  arma::mat funcs;
  auto res = algo::functional::ufpca(xt,ysmooth,0.95);
  arma::mat eigfuncs = std::get<1>(res);
  arma::mat pca = arma::join_horiz(xt.t(),mean.t(),eigfuncs.t());
  pca.save(arma::csv_name(experiments::llc::conf::train_path/"pca_prox_.csv"));

}
//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{

  mlpack::RandomSeed(SEED);

  std::filesystem::path spkr_path = "SPKR2_gsmooth95log_select_correct";
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
      /* for ( size_t j=0; j<1; j++) */
      {

        label = labels.row(j);
        error_at.resize(1,labels.n_rows);
        error.resize(1,labels.n_rows);

        X_test = inputs.cols(conf::Ntrns(i),inputs.n_cols-1);
        y_test = label.cols(conf::Ntrns(i),label.n_cols-1);

        X_train = inputs.cols(0,conf::Ntrns(i));
        y_train = label.cols(0,conf::Ntrns(i));

        size_t Ntrn = conf::Ntrns(i);
        prox_pca(y_train, Ntrn);

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
