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
/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path path = "04_07_23/comp_class_overlap"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   utils::data::classification::Dataset dataset(2,10000,2); */
/*   dataset.Generate("Hard"); */

/*   /1* dataset.Save("data.csv"); *1/ */

/*   arma::irowvec Ns = arma::regspace<arma::irowvec>(1,10,1000); */
/*   Ns.save("ns.csv",arma::csv_ascii); */
/*   size_t rep = 1000; */

/*   src::LCurve<mlpack::LogisticRegression<arma::mat>, utils::ErrorRate> LC_log(Ns,rep,true); */
/*   src::LCurve<algo::classification::LDC<DTYPE>, utils::ErrorRate> LC_ldc(Ns,rep,true); */
/*   src::LCurve<algo::classification::QDC<DTYPE>, utils::ErrorRate> LC_qdc(Ns,rep,true); */
/*   src::LCurve<algo::classification::NNC<DTYPE>, utils::ErrorRate> LC_nnc(Ns,rep,true); */

/*   LC_log.Bootstrap(dataset.inputs_,dataset.labels_); */
/*   LC_log.test_errors_.save("logit.csv", arma::csv_ascii); */

/*   LC_ldc.Bootstrap(dataset.inputs_,dataset.labels_); */
/*   LC_ldc.test_errors_.save("ldc.csv", arma::csv_ascii); */

/*   LC_qdc.Bootstrap(dataset.inputs_,dataset.labels_); */
/*   LC_qdc.test_errors_.save("qdc.csv", arma::csv_ascii); */

/*   LC_nnc.Bootstrap(dataset.inputs_,dataset.labels_); */
/*   LC_nnc.test_errors_.save("nnc.csv", arma::csv_ascii); */

/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */

int main ( int argc, char** argv )
{
  std::filesystem::path path = ".08_07_23/comp_class_overlap_log_loss";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();

  utils::data::classification::Dataset dataset(2,10000,2);
  dataset.Generate("Harder");

  /* dataset.Save("data.csv"); */

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,50);
  Ns.save("ns.csv",arma::csv_ascii);
  size_t rep = 10000;

  src::LCurve<mlpack::LogisticRegression<arma::mat>, utils::LogLoss, utils::Split> LC_log(Ns,rep,true,false,true);
  LC_log.Bootstrap(dataset.inputs_,dataset.labels_);
  LC_log.test_errors_.save("logit_harder.csv", arma::csv_ascii);

  PRINT_TIME(timer.toc());

  return 0;
}
