/**
 * @file lcgen.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef LLC_LCGEN_H 
#define LLC_LCGEN_H

namespace experiments {
namespace llc {

///////////////////////////////////////////////////////////////////////////////
//  CLASSIFICATION
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// lc_class : Read data and create learning curves for classifiers
//-----------------------------------------------------------------------------
template<class MODEL>
void lc_class ( )
{
  std::filesystem::path save_dir, filename, model_name, ids; 

  if ( typeid(MODEL) == typeid(algo::classification::NMC) )
    model_name = "nmc";
  else if ( typeid(MODEL) == typeid(algo::classification::QDC) )
    model_name = "qdc";
  else if ( typeid(MODEL) == typeid(algo::classification::LDC) )
    model_name = "ldc";
  else if ( typeid(MODEL) == typeid(algo::classification::NNC) )
    model_name = "nnc";
  else
    std::runtime_error( "Not Implemented model..." );
  
  save_dir = conf::lc_class_dir / model_name; 

  
  arma::field<std::string> header(
                        (conf::class_set.size()*conf::Nhyper*conf::Nhyper)+1);
  //arma::mat mean_train_error(conf::Ns.n_cols,
  //          (conf::class_set.size()*conf::Nhyper*conf::Nhyper));
  arma::mat mean_test_error(conf::Ns_class.n_cols,
            (conf::class_set.size()*conf::Nhyper*conf::Nhyper));
  //arma::mat std_train_error(conf::Ns.n_cols,
  //          (conf::class_set.size()*conf::Nhyper*conf::Nhyper));
  //arma::mat std_test_error(conf::Ns.n_cols,
  //          (conf::class_set.size()*conf::Nhyper*conf::Nhyper));
  header(0) = "N";

  src::classification::LCurve<MODEL, mlpack::Accuracy> 
                                            lcurve(conf::Ns_class,
                                                   int(conf::repeat));
  
  for ( size_t i=0; i<conf::class_set.size(); i++)
  {
    //#pragma omp parallel for 
    for ( size_t data_id =0; data_id<conf::Nhyper; data_id++ )
    {
      utils::data::classification::Dataset dataset;
      std::filesystem::path  dataset_file;
      dataset_file = conf::data_class_dir/conf::class_set[i]
                      /(std::to_string(data_id)+".bin");

      dataset.Load(dataset_file,true,false);
      for ( size_t model_id=0; model_id<conf::Nhyper; model_id++ )
      {
        size_t counter = conf::Nhyper*conf::Nhyper*i+(data_id*conf::Nhyper)
                                                                    +model_id;
        std::filesystem::path model_path;
        model_path = model_name/std::to_string(model_id)/conf::class_set[i]
                    /(std::to_string(data_id));
        header(counter+1) = model_path.string();
        if ( model_name == "ldc" || model_name == "qdc" )
          lcurve.Generate(dataset.inputs_,dataset.labels_,
                                          double(conf::lambda(model_id)));
        else if ( model_name == "nnc" )
          lcurve.Generate(dataset.inputs_,dataset.labels_,
                                      double(conf::k(model_id)));

        else if ( model_name == "nmc" )
          lcurve.Generate(dataset.inputs_,
                          dataset.labels_,
                          double(conf::shrink(model_id)));

        //mean_train_error.col(counter) = std::get<0>(lcurve.stats_).row(0).t();
        mean_test_error.col(counter) = std::get<1>(lcurve.stats_).row(0).t();
        //std_train_error.col(counter) = std::get<0>(lcurve.stats_).row(1).t();
        //std_test_error.col(counter) = std::get<1>(lcurve.stats_).row(1).t();
      }
    }
  }
  arma::mat N = arma::conv_to<arma::mat>::from(conf::Ns_class.t());
  //arma::mat train_m = arma::join_horiz(N, mean_train_error);
  arma::mat test_m = arma::join_horiz(N, mean_test_error);
  //arma::mat train_s = arma::join_horiz(N, std_train_error);
  //arma::mat test_s = arma::join_horiz(N, std_test_error);

  std::filesystem::path dir = conf::database_name/"classification"/model_name;
  std::filesystem::create_directories(dir);
  header.save((dir/"header.csv").string());
  //train_m.save(arma::csv_name((dir/"train_mean.csv").string(), header)); 
  test_m.save(arma::csv_name((dir/"test_mean.csv").string(), header)); 
  //train_s.save(arma::csv_name((dir/"train_std.csv").string(), header)); 
  //test_s.save(arma::csv_name((dir/"test_std.csv").string(), header)); 
}


//-----------------------------------------------------------------------------
// Generate Learning Curves: Run lc_class with desired inputs
//-----------------------------------------------------------------------------
void gen_ldc ( )
{
  PRINT("***Generating Learning Curves for LDC...")

  lc_class<algo::classification::LDC>();

  PRINT("***Generating Learning Curves for LDC...[DONE]")
}

void gen_qdc ( )
{
  PRINT("***Generating Learning Curves for QDC...")

  lc_class<algo::classification::QDC>();

  PRINT("***Generating Learning Curves for QDC...[DONE]")
}

void gen_nnc ( )
{
  PRINT("***Generating Learning Curves for NNC...")

  lc_class<algo::classification::NNC>();

  PRINT("***Generating Learning Curves for NNC...[DONE]")
}

void gen_nmc ( )
{
  PRINT("***Generating Learning Curves for NMC...")

  lc_class<algo::classification::NMC>();

  PRINT("***Generating Learning Curves for NMC...[DONE]")
}


///////////////////////////////////////////////////////////////////////////////
// REGRESSION
///////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
// lc_reg : Read data and create learning curves for regressors
//-----------------------------------------------------------------------------
template<class MODEL>
void lc_reg( )
{
  std::filesystem::path save_dir, filename, model_name, ids; 

  if ( typeid(MODEL) == typeid(algo::regression::KernelRidge
                                                  <mlpack::GaussianKernel>) )
    model_name = "gausskernelridge";
  else if ( typeid(MODEL) == typeid(algo::regression::KernelRidge
                                                  <mlpack::LaplacianKernel>) )
    model_name = "laplacekernelridge";
  else if ( typeid(MODEL) == typeid(mlpack::LinearRegression) )
    model_name = "linear";
  else
    std::runtime_error( "Not Implemented model..." );
  
  save_dir = conf::lc_reg_dir / model_name; 

  
  arma::field<std::string> header(
                        ((conf::reg_set.size())*conf::Nhyper*conf::Nhyper)+1);
  //arma::mat mean_train_error(conf::Ns.n_cols,
  //          (conf::reg_set.size()*conf::Nhyper*conf::Nhyper));
  //arma::mat std_train_error(conf::Ns.n_cols,
  //          (conf::reg_set.size()*conf::Nhyper*conf::Nhyper));
  //arma::mat std_test_error(conf::Ns.n_cols,
  //          (conf::reg_set.size()*conf::Nhyper*conf::Nhyper));
  arma::mat mean_test_error(conf::Ns_reg.n_cols,
            ((conf::reg_set.size())*conf::Nhyper*conf::Nhyper));
  header(0) = "N";

  src::regression::LCurve<MODEL, mlpack::MSE> 
                                            lcurve(conf::Ns_reg,
                                                   int(conf::repeat));
  for ( size_t i=0; i<conf::reg_set.size(); i++)
  {
    #pragma omp parallel for 
    for ( size_t data_id =0; data_id<conf::Nhyper; data_id++ )
    {

      utils::data::regression::Dataset dataset;
      std::filesystem::path  dataset_file;
      dataset_file = conf::data_reg_dir/conf::reg_set[i]
                      /(std::to_string(data_id)+".bin");

      //std::filesystem::path dataset_file = "data/regression/esinc/0.bin";

      dataset.Load(dataset_file.string(),1,1,true,false);
      arma::rowvec labels = arma::conv_to<arma::rowvec>::from
                                                            (dataset.labels_);
      for ( size_t model_id=0; model_id<conf::Nhyper; model_id++ )
      //for ( size_t model_id=0; model_id<1; model_id++ )
      {
        size_t counter = conf::Nhyper*conf::Nhyper*i+(data_id*conf::Nhyper)
                                                                    +model_id;
        std::filesystem::path model_path;
        model_path = model_name/std::to_string(model_id)/conf::reg_set[i]
                    /(std::to_string(data_id));
        header(counter+1) = model_path.string();

        lcurve.Generate(dataset.inputs_,labels,
                        double(conf::lambda(model_id)));

        mean_test_error.col(counter) = std::get<1>(lcurve.stats_).row(0).t();
        //mean_train_error.col(counter) = std::get<0>(lcurve.stats_).row(0).t();
        //std_train_error.col(counter) = std::get<0>(lcurve.stats_).row(1).t();
        //std_test_error.col(counter) = std::get<1>(lcurve.stats_).row(1).t();
      }
    }
  }
  arma::mat N = arma::conv_to<arma::mat>::from(conf::Ns_reg.t());
  arma::mat test_m = arma::join_horiz(N, mean_test_error);

  //arma::mat train_m = arma::join_horiz(N, mean_train_error);
  //arma::mat train_s = arma::join_horiz(N, std_train_error);
  //arma::mat test_s = arma::join_horiz(N, std_test_error);

  std::filesystem::path dir = conf::database_name/"regression"/model_name;
  std::filesystem::create_directories(dir);
  header.save((dir/"headers.csv").string());
  test_m.save(arma::csv_name((dir/"test_mean.csv").string(), header)); 

  //train_m.save(arma::csv_name((dir/"train_mean.csv").string(), header)); 
  //train_s.save(arma::csv_name((dir/"train_std.csv").string(), header)); 
  //test_s.save(arma::csv_name((dir/"test_std.csv").string(), header)); 
}

//-----------------------------------------------------------------------------
// lc_reg_nn : Read data and create learning curves for regressors
//-----------------------------------------------------------------------------
template<class MODEL>
void lc_reg_nn( )
{
  std::filesystem::path save_dir, filename, model_name, ids; 

  if ( typeid(MODEL) == typeid(algo::regression::NN<ens::Adam>) )
    model_name = "nn_adam";
  else if ( typeid(MODEL) == typeid(algo::regression::NN<ens::StandardSGD>) )
    model_name = "nn_sgd";
  
  save_dir = conf::lc_reg_dir / model_name; 

  
  arma::field<std::string> header(
                        ((conf::reg_set.size())*conf::Nhyper*conf::Nhyper)+1);
  //arma::mat mean_train_error(conf::Ns.n_cols,
  //          (conf::reg_set.size()*conf::Nhyper*conf::Nhyper));
  //arma::mat std_train_error(conf::Ns.n_cols,
  //          (conf::reg_set.size()*conf::Nhyper*conf::Nhyper));
  //arma::mat std_test_error(conf::Ns.n_cols,
  //          (conf::reg_set.size()*conf::Nhyper*conf::Nhyper));
  arma::mat mean_test_error(conf::Ns_reg.n_cols,
            ((conf::reg_set.size())*conf::Nhyper*conf::Nhyper));
  header(0) = "N";

  src::regression::LCurve<MODEL, mlpack::MSE> 
                                            lcurve(conf::Ns_reg,
                                                size_t(conf::repeat));
  for ( size_t i=0; i<conf::reg_set.size(); i++)
  {
    #pragma omp for 
    for ( size_t data_id =0; data_id<conf::Nhyper; data_id++ )
    {
      utils::data::regression::Dataset dataset;
      std::filesystem::path  dataset_file;
      dataset_file = conf::data_reg_dir/conf::reg_set[i]
                      /(std::to_string(data_id)+".bin");

      dataset.Load(dataset_file,1,1);

      for ( size_t model_id=0; model_id<conf::Nhyper; model_id++ )
      {
        size_t counter = conf::Nhyper*conf::Nhyper*i+(data_id*conf::Nhyper)
                                                                    +model_id;
        std::filesystem::path model_path;

        model_path = model_name/std::to_string(model_id)/conf::reg_set[i]
                    /(std::to_string(data_id));
        header(counter+1) = model_path.string();

        lcurve.Generate(dataset.inputs_,dataset.labels_,
                        conf::layer_info+model_id);

        mean_test_error.col(counter) = std::get<1>(lcurve.stats_).row(0).t();
        //mean_train_error.col(counter) = std::get<0>(lcurve.stats_).row(0).t();
        //std_train_error.col(counter) = std::get<0>(lcurve.stats_).row(1).t();
        //std_test_error.col(counter) = std::get<1>(lcurve.stats_).row(1).t();
      }

    }
  }
  arma::mat N = arma::conv_to<arma::mat>::from(conf::Ns_reg.t());
  arma::mat test_m = arma::join_horiz(N, mean_test_error);

  //arma::mat train_m = arma::join_horiz(N, mean_train_error);
  //arma::mat train_s = arma::join_horiz(N, std_train_error);
  //arma::mat test_s = arma::join_horiz(N, std_test_error);

  std::filesystem::path dir = conf::database_name/"regression"/model_name;
  std::filesystem::create_directories(dir);
  header.save((dir/"headers.csv").string());
  test_m.save(arma::csv_name((dir/"test_mean.csv").string(), header)); 

  //train_m.save(arma::csv_name((dir/"train_mean.csv").string(), header)); 
  //train_s.save(arma::csv_name((dir/"train_std.csv").string(), header)); 
  //test_s.save(arma::csv_name((dir/"test_std.csv").string(), header)); 
}
//-----------------------------------------------------------------------------
// gen_saw : Read data and create learning curves for saw shaped 
//           linear regression problem
//-----------------------------------------------------------------------------
void gen_saw(  )
{
  std::filesystem::path  save_dir, filename; 

  save_dir = conf::lc_reg_dir / conf::reg_set[4]; 
  arma::field<std::string> header(conf::Nhyper+1);
  arma::mat errors(conf::Ns_reg.n_cols,conf::Nhyper); 

  header(0) = "N";

  std::filesystem::path model_name = "saw";

  #pragma omp parallel for 
  for ( size_t model_id=0; model_id<conf::Nhyper; model_id++ )
  {
    std::filesystem::path model_path;
    model_path = model_name/std::to_string(model_id);
    header(model_id+1) = model_path.string();
    
    arma::irowvec Ns = conf::Ns_reg;
    errors.col(model_id) = create_saw(Ns,
                                       1./(model_id+conf::saw_const)).t();
                                       //1./(model_id+10)).t();
  }
  std::filesystem::path dir = conf::database_name/"regression"/model_name;
  std::filesystem::create_directories(dir);

  arma::mat N = arma::conv_to<arma::mat>::from(conf::Ns_reg.t());
  arma::mat test_m = arma::join_horiz(N, errors);
  test_m.save(arma::csv_name((dir/"test_mean.csv").string(), header)); 
  header.save((dir/"header.csv").string());
}


//-----------------------------------------------------------------------------
// gen_dgp  : Read data and create learning curves for Gaussian Processes prob.
//-----------------------------------------------------------------------------
void gen_dgp(  )
{
  std::filesystem::path  save_dir, filename, dataset_dir, directories,
                                                          dataset_file; 


  arma::field<std::string> header(conf::Nlambda_gp+1);
  arma::mat errors(conf::Ns_reg.n_cols, conf::Nlambda_gp);

  header(0) = "N";

  std::filesystem::path model_name = "dgp";

  #pragma omp parallel for 
  for ( size_t i=0; i<conf::Nlambda_gp; i++ )
  {
    arma::mat errors_(conf::Ns_reg.n_cols, conf::repeat_gp);

    std::filesystem::path model_path;

    model_path = model_name/std::to_string(i);
    header(i+1) = model_path.string();
    for ( size_t c=0; c<conf::repeat_gp; c++ )
    {
      utils::data::regression::Dataset dataset;
      std::filesystem::path dataset_file = conf::data_reg_dir /
                                           conf::reg_set[3] / std::to_string(c) 
                                            / (std::to_string(i)+".bin");
      dataset.Load(dataset_file, conf::D_gp, 1);

      arma::rowvec labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);
      src::regression::LCurve<
      algo::regression::GaussianProcess<mlpack::GaussianKernel>, mlpack::MSE>
                                        lcurve(conf::Ns_reg,
                                               int(conf::repeat_gp));

      lcurve.Generate(dataset.inputs_,labels,conf::lambda_gp(i),
                                                      2*std::sqrt(conf::D_gp));
      errors_.col(c) = std::get<1>(lcurve.stats_).row(0).t();

    }
      errors.col(i) = arma::mean(errors_,1);
  }     
  std::filesystem::path dir = conf::database_name/"regression"/model_name;
  std::filesystem::create_directories(dir);

  arma::mat N = arma::conv_to<arma::mat>::from(conf::Ns_reg.t());
  arma::mat test_m = arma::join_horiz(N, errors);
  test_m.save(arma::csv_name((dir/"test_mean.csv").string(), header)); 
  header.save((dir/"header.csv").string());

}

//-----------------------------------------------------------------------------
// Generate Learning Curves
//-----------------------------------------------------------------------------
void gen_nn_adam ( )
{
  PRINT("***Generating Learning Curves for NN...")

  lc_reg_nn<algo::regression::NN<ens::Adam>>();

  PRINT("***Generating Learning Curves for NN...[DONE]")
}

void gen_nn_sgd ( )
{
  PRINT("***Generating Learning Curves for NN...")

  lc_reg_nn<algo::regression::NN<ens::StandardSGD>>();

  PRINT("***Generating Learning Curves for NN...[DONE]")
}

void gen_lin ( )
{
  PRINT("***Generating Learning Curves for Linear...")

  lc_reg<mlpack::LinearRegression>();

  PRINT("***Generating Learning Curves for Linear...[DONE]")
}

void gen_gkr ( )
{
  PRINT("***Generating Learning Curves for GKR...")

  lc_reg<algo::regression::KernelRidge<mlpack::GaussianKernel>>();

  PRINT("***Generating Learning Curves for GKR...[DONE]")
}

void gen_lkr ( )
{
  PRINT("***Generating Learning Curves for LKR...")

  lc_reg<algo::regression::KernelRidge<mlpack::LaplacianKernel>>();

  PRINT("***Generating Learning Curves for LKR...[DONE]")
}

//-----------------------------------------------------------------------------
// gen_reg: Run the regression problems
//-----------------------------------------------------------------------------
void gen_reg ( )
{
  PRINT("***Generating Learning Curves for regression...")

  gen_gkr();
  gen_nn_adam();
  gen_lin();
  gen_saw();
  gen_dgp();


  PRINT("***Generating Learning Curves for regression...[DONE]")
}

//-----------------------------------------------------------------------------
// gen_clas : Run lc_class with desired inputs
//-----------------------------------------------------------------------------
void gen_clas( )
{
  PRINT("***Generating Learning Curves for classification...")

  gen_nmc();
  gen_ldc();
  gen_qdc();
  gen_nnc();

  PRINT("***Generating Learning Curves for classification...[DONE]")
}

//-----------------------------------------------------------------------------
// lcgen: Run lc_class with desired inputs
//-----------------------------------------------------------------------------
void lcgen( )
{
  gen_clas();
  gen_reg();
}



//-----------------------------------------------------------------------------
// normalize : Normalize database
//-----------------------------------------------------------------------------
void normalize(  )
{

  auto normy = [](const std::filesystem::path& path) 
  {
    arma::mat train, test, ytrn, ytst; 
    arma::field<std::string> header_train, header_test; 
    train.load(arma::csv_name(path/"train_original.csv",
                                                      header_train));   
    test.load(arma::csv_name(path/"test_original.csv",
                                                      header_train));   
    arma::vec X = train.col(0);
    ytrn = train.cols(1,train.n_cols);
    ytst = test.cols(1,train.n_cols);
    arma::vec temp;
    for (size_t i=0; i<ytrn.n_cols; i++)
    {
      temp = ytrn.col(i);
      ytrn.col(i) = ytrn.col(i)/arma::trapz(X,temp);
      temp = ytst.col(i);
      ytst.col(i) = ytst.col(i)/arma::trapz(X,temp);
    }
    arma::mat save_train = arma::join_horiz(X,ytrn);
    arma::mat save_test = arma::join_horiz(X,ytst);
    save_train.save(arma::csv_name(conf::train_path_class/"train.csv",
                                                      header_train));
    save_train.save(arma::csv_name(conf::train_path_class/"train.csv",
                                                      header_train));

  };

  normy(conf::train_path_class);
  normy(conf::train_path_reg);
}

//-----------------------------------------------------------------------------
// split : Split database
//-----------------------------------------------------------------------------
void split ( )
{
  
  auto splity = [](const std::filesystem::path& path) 
  {
    auto splitted = utils::BulkLoadSplit2(path,0.2,false);
    arma::mat train = std::get<0>(splitted);
    arma::mat test = std::get<1>(splitted);
    utils::Save(path/"train_original.csv", train,false);
    utils::Save(path/"test_original.csv", test,false);
  };

  splity(conf::database_name/"classification");
  splity(conf::database_name/"regression");

}

} // namespace llc
} // namespace experiments
#endif 

