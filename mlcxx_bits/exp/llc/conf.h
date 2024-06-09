/**
 *
 * @file conf.h
 * @author Ozgur Taylan Turan
 *
 * Configuration variables for llc experiments
 */

#ifndef LLC_CONF_H
#define LLC_CONF_H

namespace experiments {
namespace llc {
namespace conf {

  const std::filesystem::path root = ".llc-paper";
  const std::filesystem::path database_name = root/"LCDB";

  const size_t Nhyper = 20;

  const std::vector<std::string> class_set = 
                                           {"gaussian","banana","rdip", "ddip"};

  const std::vector<std::string> reg_set = 
                           {"elin","esinc","esine", "dgp", "saw"};

  const std::filesystem::path data_class_dir = root/"data/classification";
  const std::filesystem::path data_reg_dir = root/"data/regression";


  //----- dataset.h
  
  const size_t N = 10000;
  // -- regression
  const size_t D = 1;
  const size_t Neps = Nhyper;
  const size_t Nd = 20;

  const size_t D_gp = 10;
  const size_t Nlambda_gp = Nhyper; 
  const size_t repeat_gp = 10;

  const arma::vec lambda_gp_teach = arma::linspace(1e-6, 1., Nlambda_gp);

  const arma::vec eps = arma::linspace(0.,1., Neps);

  // -- classification
  const size_t Ddip = 1;
  const size_t Dc = 2;
  const size_t Nc = 2;
  const size_t Nr = Nhyper;
  const size_t Ndelta = Nhyper;
  const arma::vec r = arma::linspace(1,100,Nr);
  const arma::vec mean = {0,0}; 

  const arma::vec delta = arma::linspace(0.1,5,Ndelta);


  //----- dataset.h
  
  //----- lcgen.h
  
  const size_t repeat = 100;

  const arma::rowvec lambda = arma::linspace<arma::rowvec>(1e-5,1.,Nhyper);
  const arma::rowvec shrink = arma::linspace<arma::rowvec>(0,10,Nhyper);
  //const arma::rowvec lambda_dc = arma::linspace<arma::rowvec>(0.,1.e-5,Nhyper);
  const arma::Row<size_t>  k = arma::regspace<arma::Row<size_t>>(1,1,Nhyper);

  const size_t init_layer = 5;

  const arma::Row<size_t> layer_info = {conf::init_layer,conf::init_layer};

  const size_t saw_const = 2.;

  const size_t Ns_start1 = 2;
  const size_t Ns_end1 = 100;
  const size_t Ns_step1 = 1;

  const size_t Ns_start2 = 2;
  const size_t Ns_end2 = 100;
  const size_t Ns_step2 = 1;


  const arma::irowvec Ns_reg = arma::regspace<arma::irowvec>(Ns_start1,
                                                         Ns_step1,Ns_end1);

  const arma::irowvec Ns_class = arma::regspace<arma::irowvec>(Ns_start2,
                                                         Ns_step2,Ns_end2);

  const arma::irowvec Ns = Ns_class; 
                          

  const arma::vec lambda_gp = arma::linspace(1e-6, 1e-2, Nlambda_gp);

  const std::filesystem::path lc_reg_dir = root/"regression";
  const std::filesystem::path lc_class_dir = 
                                       root/"classification";

  //----- lcgen.h

  //----- curve_fit.h & spkr_fit.h

  const std::vector<std::string> keys = { "ldc","qdc","nmc","saw",
                                          "gausskernelridge"};

  const size_t idx = Ns.n_cols-1;
  const size_t restart_opt = 100;

  const arma::rowvec Ntrns = {10,25,50};
  /* const arma::rowvec Ntrns = {50}; */
  
  const std::string type_func = "pca";

  const double valid = 0.2; const size_t ngrid = 20;
  /* const arma::rowvec lambdas = arma::linspace<arma::rowvec>(1.,10,ngrid); */
  const arma::rowvec lambdas = arma::logspace<arma::rowvec>(0,2,ngrid);
  /* const arma::rowvec lambdas = arma::linspace<arma::rowvec>(1e-6,1,ngrid); */
  //const arma::Row<size_t> npcs = arma::regspace<arma::Row<size_t>>
  //                                                              (1,1,ngrid);
  const arma::Row<size_t> npcs = arma::regspace<arma::Row<size_t>>
                                                                (1,2,2*ngrid);

  /* const arma::rowvec ls = arma::linspace<arma::rowvec>(1.,10.,ngrid); */
  const arma::rowvec ls = arma::logspace<arma::rowvec>(1,2,ngrid);
  
  const size_t n_cluster = 10;

  const size_t n_closest = 100;


  const std::filesystem::path train_root = root/"LCDB_0_12";
  const std::filesystem::path test_root = root/"LCDB_0_12";
  //const std::filesystem::path train_file = "train_original.csv";
  //const std::filesystem::path test_file = "test_original.csv";
  //const std::filesystem::path train_file = "train.csv";
  const std::filesystem::path train_file = "clusters_.csv";
  const std::filesystem::path test_file = "test.csv";
  const std::filesystem::path at_file = "last.csv";
  const std::filesystem::path err_file = "error.csv";
  const std::filesystem::path pred_file = "predictions.csv";

  const std::vector<std::string> run_keys = { "all", "all", "qdc","ldc","nmc",
                                              "saw", "gausskernelridge"};
  const size_t expid = 1;
  const std::string which = run_keys[expid];
  const bool subset = std::count(keys.begin(), keys.end(), which);
    
  
  const std::vector<std::filesystem::path> train_paths = 
  { 
    train_root/"classification",                    // "0"
    train_root/"regression",                        // "1"
    /* train_root/"classification",                    // "2" only qdc */
    /* train_root/"classification",                    // "3" only ldc */
    /* train_root/"classification",                    // "4" only nmc */
    /* train_root/"regression",                        // "5" only saw */ 
    /* train_root/"regression",                        // "6" only kr */
  };

  const std::vector<std::filesystem::path> test_paths = 
  { 
    // FINAL
    test_root/"classification",                    // "0"
    test_root/"regression",                        // "1"
    /* test_root/"classification",                    // "2" only qdc */   
    /* test_root/"classification",                    // "3" only ldc */   
    /* test_root/"classification",                    // "4" only nmc */   
    /* test_root/"regression",                        // "5" only saw */   
    /* test_root/"regression",                        // "6" only kr */
  };

  const std::filesystem::path train_path = train_paths[expid];
  const std::filesystem::path test_path = test_paths[expid];
  const std::filesystem::path test_path_class = test_root/"classification";
  const std::filesystem::path test_path_reg = test_root/"regression";

  const std::filesystem::path train_path_class = train_root/"classification";
  const std::filesystem::path train_path_reg = train_root/"regression";

  const std::filesystem::path exp_id = "EXP-"+std::to_string(expid);
  

  
} // namespace conf
} // namespace llc
} // namespace experiments

#endif

