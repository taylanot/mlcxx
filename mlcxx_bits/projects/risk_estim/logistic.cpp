/**
 * @file logistic .cpp
 * @author Ozgur Taylan Turan
 *
 * Looking at the Logistric Regression of mlpack
 */

#include <headers.h>

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  std::filesystem::path path = ".23_07_23/synth";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();

  utils::data::classification::Dataset dataset(2,10000,2);
  dataset.Generate("Harder");

  /* dataset.Save("data.csv"); */

  arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,700);
  size_t rep = 100;

  /* src::LCurve<mlpack::LogisticRegression<arma::mat>, utils::ErrorRate> */ 
  src::LCurve<mlpack::LogisticRegression<arma::mat>, utils::LogLoss> 
    LC_log(Ns,rep,true,false,true);

  LC_log.Bootstrap(dataset.inputs_,dataset.labels_);
  LC_log.test_errors_.save("logloss.csv", arma::csv_ascii);

  PRINT_TIME(timer.toc());

  return 0;
}

/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path path = ".08_07_23/comp_class_overlap_log_loss"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   utils::data::classification::Dataset dataset(2,10000,2); */
/*   dataset.Generate("Harder"); */

/*   /1* dataset.Save("data.csv"); *1/ */

/*   arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,50); */
/*   Ns.save("ns.csv",arma::csv_ascii); */
/*   size_t rep = 10000; */

/*   src::LCurve<mlpack::LogisticRegression<arma::mat>, utils::LogLoss, utils::Split> LC_log(Ns,rep,true,false,true); */
/*   LC_log.Bootstrap(dataset.inputs_,dataset.labels_); */
/*   LC_log.test_errors_.save("logit_harder.csv", arma::csv_ascii); */

/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path path = ".23_07_23/iris"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   utils::data::classification::Dataset dataset; */
/*   dataset.Load(MLCXX_ROOT/"datasets/iris.csv",true,true); */

/*   arma::irowvec Ns = arma::regspace<arma::irowvec>(2,20,150); */
/*   size_t rep = 100000; */

/*   src::LCurve<algo::classification::MultiClass<mlpack::LogisticRegression<arma::mat>>, */
/*               utils::LogLoss, */
/*               utils::StratifiedSplit> LC_log(Ns,rep,true,false,true); */
/*   LC_log.Bootstrap(dataset.inputs_,dataset.labels_); */
/*   LC_log.test_errors_.save("boot.csv", arma::csv_ascii); */
/*   LC_log.Additive(dataset.inputs_,dataset.labels_); */
/*   LC_log.test_errors_.save("add.csv", arma::csv_ascii); */
/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path path = ".23_07_23/iris"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   utils::data::classification::Dataset dataset; */
/*   dataset.Load(MLCXX_ROOT/"datasets/iris.csv",true,true); */

/*   arma::irowvec Ns = arma::regspace<arma::irowvec>(2,20,150); */
/*   size_t rep = 100000; */

/*   src::LCurve<algo::classification::MultiClass<mlpack::LogisticRegression<arma::mat>>, */
/*               utils::LogLoss, */
/*               utils::StratifiedSplit> LC_log(Ns,rep,true,false,true); */
/*   LC_log.Bootstrap(dataset.inputs_,dataset.labels_); */
/*   LC_log.test_errors_.save("boot.csv", arma::csv_ascii); */
/*   LC_log.Additive(dataset.inputs_,dataset.labels_); */
/*   LC_log.test_errors_.save("add.csv", arma::csv_ascii); */
/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */


/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path path = ".23_07_23/mnist"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   utils::data::classification::Dataset trainset,testset; */

/*   trainset.Load(MLCXX_ROOT/"datasets/mnist-dataset/mnist_train.csv",false,true,false); */
/*   testset.Load(MLCXX_ROOT/"datasets/mnist-dataset/mnist_test.csv",false,true,false); */

/*   trainset.inputs_ /= 255; */
/*   testset.inputs_ /= 255; */


/*   arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,10); */
/*   size_t rep = 100; */

/*   src::LCurve<algo::classification::MultiClass<mlpack::LogisticRegression<arma::mat>>, */
/*               utils::LogLoss, */
/*               /1* mlpack::Accuracy, *1/ */
/*               utils::StratifiedSplit> LC_log(Ns,rep,true,false,true); */
/*   LC_log.Split(trainset,testset); */
/*   LC_log.test_errors_.save("split_smaller.csv", arma::csv_ascii); */
/*   /1* /2* LC_log.Bootstrap(trainset.inputs_,trainset.labels_); *2/ *1/ */
/*   /1* /2* LC_log.test_errors_.save("boot.csv", arma::csv_ascii); *2/ *1/ */

/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */
