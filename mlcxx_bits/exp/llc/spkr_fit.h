/**
 * @file spkr_fit.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef LLC_SPKRFIT_H 
#define LLC_SPKRFIT_H

namespace experiments {
namespace llc {




//=============================================================================
// FUNCS : Class for reading and storing training functionals given number of 
//        functionals...
//=============================================================================
class FUNCS 
{
  public:
  
  FUNCS ( const size_t& M ) : M_(M)
  {
    //arma::mat data = utils::Load(conf::train_path/conf::train_file, true);
    arma::field<std::string> header;
    arma::mat data;
    data.load(arma::csv_name(conf::train_path/conf::train_file, header));   
    arma::inplace_trans(data);
    X_ = data.row(0);
    y_ = data.rows(1,data.n_rows-1);
  }
  
  arma::mat Predict ( const arma::mat& X,
                      const std::string& type = conf::type_func ) 
  {
    arma::mat labels;
    arma::mat temp;

    arma::uvec idx, tempid;
    idx.resize(X.n_cols);
    for ( size_t i=0; i < X.n_cols; i++)
    {
      tempid = arma::find(X_ == double(X(0,i)));
      idx(i) = tempid(0);
    }
    if ( type == "raw" )
    {
      arma::uvec fid = arma::regspace<arma::uvec>(0,1,M_-1);
      temp = y_.rows(fid); labels = temp.cols(idx);
    }
    else if ( type == "pca" )
    {
      auto results = algo::functional::ufpca(X_, y_, size_t(M_));
      temp =  std::get<1>(results);
      labels = temp.cols(idx);
    }
    else
      throw std::runtime_error("No other option!");

    return labels;
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


    temp = arma::mean(y_,0); labels = temp.cols(idx);

    return labels;
  }

  private:
  arma::mat X_;
  arma::mat y_;
  size_t M_;
};


class SFUNCS
{
  public:
  
  SFUNCS ( const size_t& M, std::string which=conf::which ) : M_(M),
                                                              which_(which)
  {
    arma::field<std::string> header;
    arma::mat data;
    data.load(arma::csv_name(conf::train_path/conf::train_file, header));   
    arma::inplace_trans(data);
    X_ = data.row(0);
    y_ = data.rows(select(header, which_));

  }
  
  arma::mat Predict ( const arma::mat& X,
                      const std::string& type = conf::type_func ) 
  {
    arma::mat labels;
    arma::mat temp;

    arma::uvec idx, tempid;
    idx.resize(X.n_cols);
    for ( size_t i=0; i < X.n_cols; i++)
    {
      // What is this doing now?
      tempid = arma::find(X_ == double(X(0,i)));
      idx(i) = tempid(0);
    }
    if ( type == "raw" )
    {
      arma::uvec fid = arma::regspace<arma::uvec>(0,1,M_-1);
      temp = y_.rows(fid); labels = temp.cols(idx);
    }
    else if ( type == "pca" )
    {
      auto results = algo::functional::ufpca(X_, y_, size_t(M_));
      temp =  std::get<1>(results);
      labels = temp.cols(idx);
    }
    else
      throw std::runtime_error("No other option!");

    return labels;
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


    temp = arma::mean(y_,0); labels = temp.cols(idx);

    return labels;
  }

  private:
  arma::mat X_;
  arma::mat y_;
  size_t M_;
  std::string which_;
};

class SFUNCS1
{
  public:
  
  SFUNCS1 ( const size_t& M, std::string which="saw" ) : M_(M),
                                                              which_(which)
  {
    arma::field<std::string> header;
    arma::mat data;
    data.load(arma::csv_name(conf::train_path_reg/conf::train_file, header));   
    arma::inplace_trans(data);
    X_ = data.row(0);
    y_ = data.rows(select2(header, which_));

  }
  
  arma::mat Predict ( const arma::mat& X,
                      const std::string& type = conf::type_func ) 
  {
    arma::mat labels;
    arma::mat temp;

    arma::uvec idx, tempid;
    idx.resize(X.n_cols);
    for ( size_t i=0; i < X.n_cols; i++)
    {
      tempid = arma::find(X_ == double(X(0,i)));
      idx(i) = tempid(0);
    }
    if ( type == "raw" )
    {
      arma::uvec fid = arma::regspace<arma::uvec>(0,1,M_-1);
      temp = y_.rows(fid); labels = temp.cols(idx);
    }
    else if ( type == "pca" )
    {
      auto results = algo::functional::ufpca(X_, y_, size_t(M_));
      temp =  std::get<1>(results);
      labels = temp.cols(idx);
    }
    else
      throw std::runtime_error("No other option!");

    return labels;
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


    temp = arma::mean(y_,0); labels = temp.cols(idx);

    return labels;
  }

  private:
  arma::mat X_;
  arma::mat y_;
  size_t M_;
  std::string which_;
};

class SFUNCS2
{
  public:
  
  SFUNCS2 ( const size_t& M, std::string which= "ddip" ) : M_(M),
                                                              which_(which)
  {
    arma::field<std::string> header;
    arma::mat data;
    data.load(arma::csv_name(conf::train_path_class/conf::train_file, header));   
    arma::inplace_trans(data);
    X_ = data.row(0);

    y_ = data.rows(select2(header, which_));

  }
  
  arma::mat Predict ( const arma::mat& X,
                      const std::string& type = conf::type_func ) 
  {
    arma::mat labels;
    arma::mat temp;

    arma::uvec idx, tempid;
    idx.resize(X.n_cols);
    for ( size_t i=0; i < X.n_cols; i++)
    {
      tempid = arma::find(X_ == double(X(0,i)));
      idx(i) = tempid(0);
    }
    if ( type == "raw" )
    {
      arma::uvec fid = arma::regspace<arma::uvec>(0,1,M_-1);
      temp = y_.rows(fid); labels = temp.cols(idx);
    }
    else if ( type == "pca" )
    {
      auto results = algo::functional::ufpca(X_, y_, size_t(M_));
      temp =  std::get<1>(results);
      labels = temp.cols(idx);
    }
    else
      throw std::runtime_error("No other option!");

    return labels;
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


    temp = arma::mean(y_,0); labels = temp.cols(idx);

    return labels;
  }

  private:
  arma::mat X_;
  arma::mat y_;
  size_t M_;
  std::string which_;
};
//-----------------------------------------------------------------------------
// subset_run_spkr2_fit : Run SPKR fitting for subset
//-----------------------------------------------------------------------------
void subset_run_spkr2_fit ( )
{
  std::filesystem::path spkr_path = "SPKR2";

  arma::field<std::string> header;
  arma::mat data;
  data.load(arma::csv_name(conf::test_path/conf::test_file, header));   
  arma::inplace_trans(data);
  
  arma::mat inputs = data.row(0);
  //arma::mat labels = data.rows(1,data.n_rows-1);
  arma::uvec ids = select(header, conf::which);
  arma::field<std::string> sheader(1,ids.n_elem);
  arma::mat labels = data.rows(ids);
  for(size_t i=0 ; i<ids.n_elem; i++)
  {
    sheader(0,i) = header(ids[i]);
  }
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
        label = labels.row(j);
        error_at.resize(1,labels.n_rows);
        error.resize(1,labels.n_rows);

        X_test = inputs.cols(conf::Ntrns(i),inputs.n_cols-1);
        y_test = label.cols(conf::Ntrns(i),label.n_cols-1);

        X_train = inputs.cols(0,conf::Ntrns(i));
        y_train = label.cols(0,conf::Ntrns(i));

        mlpack::hpt::HyperParameterTuner
         <algo::regression::SemiParamKernelRidge2
                  <mlpack::GaussianKernel,SFUNCS>,
          mlpack::cv::MSE,
          mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train);

        //double best_lambda=0.1;
        //double  best_l=0.1;
        //size_t best_npc=1;

        double best_lambda, best_l;
        size_t best_npc;
        std::tie(best_lambda, best_npc, best_l) = 
                          hpt.Optimize(conf::lambdas,conf::npcs,conf::ls);

        algo::regression::SemiParamKernelRidge2<mlpack::GaussianKernel,SFUNCS> 
                  model(X_train, y_train, best_lambda, best_npc, best_l );

        PRINT(conf::idx);
        PRINT(arma::size(inputs));
        X_test_at(0,0) = inputs(conf::idx);
        y_test_at(0) = label(conf::idx);

        error_at(j) = model.ComputeError(X_test_at,y_test_at);
        error(j) = model.ComputeError(inputs,label);
        arma::rowvec temp_y;
        model.Predict(inputs,temp_y);
        predictions.row(j) = temp_y; 
      }
      arma::inplace_trans(predictions);


      error_at.save(arma::csv_name(path_clean/conf::at_file,
                                   sheader.cols(0,labels.n_rows-1)));
      error.save(arma::csv_name(path_clean/conf::err_file,
                                   sheader.cols(0,labels.n_rows-1)));
      predictions.save(arma::csv_name(path_clean/conf::pred_file,
                                   sheader.cols(0,labels.n_rows-1)));
    }
  }
}

//-----------------------------------------------------------------------------
// run_spkr2_fit : Run SPKR fitting
//-----------------------------------------------------------------------------
void run_spkr2_fit ( )
{
  std::filesystem::path spkr_path = "SPKR2";

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
      {
        label = labels.row(j);
        error_at.resize(1,labels.n_rows);
        error.resize(1,labels.n_rows);

        X_test = inputs.cols(conf::Ntrns(i),inputs.n_cols-1);
        y_test = label.cols(conf::Ntrns(i),label.n_cols-1);

        X_train = inputs.cols(0,conf::Ntrns(i));
        y_train = label.cols(0,conf::Ntrns(i));

        mlpack::hpt::HyperParameterTuner
         <algo::regression::SemiParamKernelRidge2
                  <mlpack::GaussianKernel,FUNCS>,
          mlpack::cv::MSE,
          mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train);

        double best_lambda, best_l;
        size_t best_npc;
        
        std::tie(best_lambda, best_npc, best_l) = 
                          hpt.Optimize(conf::lambdas,conf::npcs,conf::ls);

        algo::regression::SemiParamKernelRidge2<mlpack::GaussianKernel,
                                               FUNCS> 
                  model(X_train, y_train, best_lambda, best_npc, best_l );


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
}

//-----------------------------------------------------------------------------
// run_spkr2_fit_before: Run SPKR fitting for selected curves
//-----------------------------------------------------------------------------
void run_spkr2_fit_before( )
{
  size_t id_saw = 0;
  size_t id_ddip = 41;
  std::filesystem::create_directories("results/ddip");
  std::filesystem::create_directories("results/saw");

  arma::field<std::string> header_class;
  arma::mat data_class;
  data_class.load(arma::csv_name(conf::test_path_class/conf::test_file,
                  header_class));   
  arma::inplace_trans(data_class);
  arma::uvec ids_class = select2(header_class,"ddip");

  arma::mat inputs_class = data_class.row(0);
  arma::mat labels_class = data_class.rows(ids_class.row(id_ddip));
  
  arma::field<std::string> header_reg;
  arma::mat data_reg;
  data_reg.load(arma::csv_name(conf::test_path_reg/conf::test_file,header_reg));   
  arma::inplace_trans(data_reg);
  arma::uvec ids_reg = select2(header_reg,"saw");

  arma::mat inputs_reg = data_reg.row(0);
  arma::mat labels_reg = data_reg.rows(ids_reg.row(id_saw));


  arma::mat X_train, X_test; arma::rowvec y_train, y_test;

  for ( size_t i=0; i<conf::Ntrns.n_cols; i++ )
  {
    arma::rowvec label;

    arma::mat predictions_reg(labels_reg.n_rows,conf::Ns.n_elem);
    arma::mat predictions_class(labels_class.n_rows,conf::Ns.n_elem);

    for ( size_t j=0; j<labels_reg.n_rows; j++)
    {
      label = labels_reg.row(j);

      X_train = inputs_reg.cols(0,conf::Ntrns(i));
      y_train = labels_reg.cols(0,conf::Ntrns(i));

      mlpack::hpt::HyperParameterTuner
       <algo::regression::SemiParamKernelRidge2
                <mlpack::GaussianKernel,FUNCS>,
        mlpack::cv::MSE,
        mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train);

      double best_lambda, best_l;
      size_t best_npc;
      
      std::tie(best_lambda, best_npc, best_l) = 
                        hpt.Optimize(conf::lambdas,conf::npcs,conf::ls);

      algo::regression::SemiParamKernelRidge2<mlpack::GaussianKernel,
                                             FUNCS> 
                model(X_train, y_train, best_lambda, best_npc, best_l );

      arma::rowvec temp_y;
      model.Predict(inputs_reg,temp_y);
      predictions_reg.row(j) = temp_y; 
    }
    for ( size_t j=0; j<labels_class.n_rows; j++)
    {
      label = labels_class.row(j);

      X_train = inputs_class.cols(0,conf::Ntrns(i));
      y_train = labels_class.cols(0,conf::Ntrns(i));

      mlpack::hpt::HyperParameterTuner
       <algo::regression::SemiParamKernelRidge2
                <mlpack::GaussianKernel,FUNCS>,
        mlpack::cv::MSE,
        mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train);

      double best_lambda, best_l;
      size_t best_npc;
      
      std::tie(best_lambda, best_npc, best_l) = 
                        hpt.Optimize(conf::lambdas,conf::npcs,conf::ls);

      algo::regression::SemiParamKernelRidge2<mlpack::GaussianKernel,
                                             FUNCS> 
                model(X_train, y_train, best_lambda, best_npc, best_l );

      arma::rowvec temp_y;
      model.Predict(inputs_class,temp_y);
      predictions_class.row(j) = temp_y; 
    }
    predictions_reg.save("results/saw/before.csv",arma::csv_ascii);
    predictions_class.save("results/ddip/before.csv",arma::csv_ascii);
  }
}

//-----------------------------------------------------------------------------
// run_spkr2_fit_after: Run SPKR fitting for selected curves
//-----------------------------------------------------------------------------
void run_spkr2_fit_after( )
{
  size_t id_saw = 0;
  size_t id_ddip = 41;
  std::filesystem::create_directories("results/ddip");
  std::filesystem::create_directories("results/saw");

  arma::field<std::string> header_class;
  arma::mat data_class;
  data_class.load(arma::csv_name(conf::test_path_class/conf::test_file,
                  header_class));   
  arma::inplace_trans(data_class);
  arma::uvec ids_class = select2(header_class,"ddip");

  arma::mat inputs_class = data_class.row(0);
  arma::mat labels_class = data_class.rows(ids_class.row(id_ddip));
  
  arma::field<std::string> header_reg;
  arma::mat data_reg;
  data_reg.load(arma::csv_name(conf::test_path_reg/conf::test_file,header_reg));   
  arma::inplace_trans(data_reg);
  arma::uvec ids_reg = select2(header_reg,"saw");

  arma::mat inputs_reg = data_reg.row(0);
  arma::mat labels_reg = data_reg.rows(ids_reg.row(id_saw));


  arma::mat X_train, X_test; arma::rowvec y_train, y_test;

  for ( size_t i=0; i<conf::Ntrns.n_cols; i++ )
  {
    arma::rowvec label;

    arma::mat predictions_reg(labels_reg.n_rows,conf::Ns.n_elem);
    arma::mat predictions_class(labels_class.n_rows,conf::Ns.n_elem);

    for ( size_t j=0; j<labels_reg.n_rows; j++)
    {
      label = labels_reg.row(j);

      X_train = inputs_reg.cols(0,conf::Ntrns(i));
      y_train = labels_reg.cols(0,conf::Ntrns(i));

      mlpack::hpt::HyperParameterTuner
       <algo::regression::SemiParamKernelRidge2
                <mlpack::GaussianKernel,SFUNCS1>,
        mlpack::cv::MSE,
        mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train);

      double best_lambda, best_l;
      size_t best_npc;
      
      std::tie(best_lambda, best_npc, best_l) = 
                        hpt.Optimize(conf::lambdas,conf::npcs,conf::ls);

      algo::regression::SemiParamKernelRidge2<mlpack::GaussianKernel,
                                             SFUNCS1> 
                model(X_train, y_train, best_lambda, best_npc, best_l );

      arma::rowvec temp_y;
      model.Predict(inputs_reg,temp_y);
      predictions_reg.row(j) = temp_y; 
    }
    predictions_reg.save("results/saw/after.csv",arma::csv_ascii);
    for ( size_t j=0; j<labels_class.n_rows; j++)
    {
      label = labels_class.row(j);

      X_train = inputs_class.cols(0,conf::Ntrns(i));
      y_train = labels_class.cols(0,conf::Ntrns(i));

      mlpack::hpt::HyperParameterTuner
       <algo::regression::SemiParamKernelRidge2
                <mlpack::GaussianKernel,SFUNCS2>,
        mlpack::cv::MSE,
        mlpack::cv::SimpleCV> hpt(conf::valid, X_train, y_train);

      double best_lambda, best_l;
      size_t best_npc;
      
      std::tie(best_lambda, best_npc, best_l) = 
                        hpt.Optimize(conf::lambdas,conf::npcs,conf::ls);

      PRINT("NOT")
      algo::regression::SemiParamKernelRidge2<mlpack::GaussianKernel,
                                             SFUNCS2> 
                model(X_train, y_train, best_lambda, best_npc, best_l );

      arma::rowvec temp_y;
      model.Predict(inputs_class,temp_y);
      predictions_class.row(j) = temp_y; 
    }
    predictions_class.save("results/ddip/after.csv",arma::csv_ascii);
  }
}

} //namespace llc
} //namespacee experiments

#endif
