/**
 * @file cluster_spkr.cpp
 * @author Ozgur Taylan Turan
 *
 */

#include <headers.h>

using namespace experiments::llc;



class CLUSTERREAD
{
  public:
  
  CLUSTERREAD( const size_t& M=0 ) : M_(M)
  {
    arma::mat data;
    data.load(conf::train_path/conf::train_file);   
    arma::inplace_trans(data);
    X_ = data.row(0);
    M_ = data.n_rows-1;
    y_ = data.rows(1,data.n_rows-1);
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

  private:
  arma::mat X_;
  arma::mat y_;
  size_t M_;
};

void cluster( )
{
  arma::mat train_data;
  arma::field<std::string> train_header;
  train_data.load(arma::csv_name(conf::train_path_class/"train.csv",
          train_header));

  arma::mat means;
  arma::mat ys = train_data.cols(1,train_data.n_cols-1);
  arma::kmeans(means,ys,100, arma::random_subset, 10000, false);
  means = arma::join_horiz(train_data.col(0),means);
  means.save(
      arma::csv_name(experiments::llc::conf::train_path/"clusters_.csv"));
}

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  mlpack::RandomSeed(SEED);

  cluster();  

  std::filesystem::path spkr_path = "SPKR2_cluster"+
                                      std::to_string(conf::n_cluster);
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

        mlpack::hpt::HyperParameterTuner
        <algo::regression::SemiParamKernelRidge
                  <mlpack::GaussianKernel,CLUSTERREAD>,
          mlpack::cv::MSE,
          mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train);
        double best_lambda, best_l;
        size_t best_npc = 0;
        
        std::tie(best_lambda, best_l) = 
                          hpt.Optimize(conf::lambdas,mlpack::Fixed(best_npc),conf::ls);

        algo::regression::SemiParamKernelRidge<mlpack::GaussianKernel,
                                               CLUSTERREAD> 
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
