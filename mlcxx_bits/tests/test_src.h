/**
 * @file test_src.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_SRC_H 
#define TEST_SRC_H

TEST_SUITE("LEARNINGCURVES") {
  TEST_CASE("REGRESSION")
  {
    SUBCASE("NOHPT")
    {
      int D, Ntrn;  D=1; Ntrn=1000; 
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      utils::data::regression::Dataset dataset(D, Ntrn);

      dataset.Generate(a, p, "Linear", eps);
      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      int repeat = 10; 

      src::regression::LCurve<mlpack::LinearRegression,
             mlpack::MSE> lcurve(Ns,repeat);
      
      lcurve.Generate(inputs, labels, 0., 1.);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }

    SUBCASE("VARIABLE")
    {
      int D, Ntrn;  D=1; Ntrn=1000; 
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      utils::data::regression::Dataset dataset(D, Ntrn);
      dataset.Generate(a, p, "Linear", eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      arma::irowvec repeat = arma::regspace<arma::irowvec>(2,1,5);

      src::regression::VariableLCurve<mlpack::LinearRegression,
             mlpack::MSE> lcurve(Ns,repeat);
      
      lcurve.Generate(inputs, labels, 0., 1.);

     // CHECK ( arma::mean(arma::mean(
     //                (lcurve.test_errors_[0] - lcurve.train_errors_[0]))) > 0 );
    }

    SUBCASE("HPT")
    {
      int D, Ntrn;  D=1; Ntrn=1000; 
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      utils::data::regression::Dataset dataset(D, Ntrn);

      dataset.Generate(a, p, "Linear", eps);
      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,20);
      int repeat = 10; 
      src::regression::LCurve_HPT<mlpack::LinearRegression,
                 mlpack::MSE,
                 mlpack::SimpleCV> lcurve(Ns,repeat,0.2);

      arma::rowvec lambdas = arma::linspace<arma::rowvec>(0.,1.,100);
      lcurve.Generate(inputs, labels, lambdas);

      CHECK ( arma::mean(arma::mean(
                     (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }
  } 

  TEST_CASE("CLASSIFICATION")
  {
    SUBCASE("NOHPT")
    {
      int D, N, Nc;
      D = 1; N = 1000; Nc = 2;
      utils::data::classification::Dataset dataset(D, N, Nc);
      std::string type = "Hard";
      dataset.Generate(type);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      int repeat = 10; 
      auto inputs = dataset.inputs_;
      auto labels = dataset.labels_;
      src::classification::LCurve<algo::classification::NMC,
             mlpack::MSE> lcurve(Ns,repeat);
      
      lcurve.Generate(inputs, labels);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) >= 0 );
    }
  }
}

#endif

