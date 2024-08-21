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
    SUBCASE("BOOT")
    {
      int D, Ntrn;  D=1; Ntrn=10; 
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      data::regression::Dataset dataset(D, Ntrn);

      dataset.Generate(a, p, "Linear", eps);
      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      int repeat = 10; 

      src::LCurve<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE>
                                                              lcurve(Ns,repeat);
      
      lcurve.Bootstrap(inputs, labels, 0., 1.);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }
    SUBCASE("ADD")
    {
      int D, Ntrn;  D=1; Ntrn=10; 
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      data::regression::Dataset dataset(D, Ntrn);

      dataset.Generate(a, p, "Linear", eps);
      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      int repeat = 10; 

      src::LCurve<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> lcurve(Ns,repeat);
      
      lcurve.Additive(inputs, labels, 0., 1.);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }
    SUBCASE("SPLIT")
    {
      int D, Ntrn, Ntst;  D=1; Ntrn=20; Ntst=10;
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      data::regression::Dataset trainset(D, Ntrn);
      data::regression::Dataset testset(D, Ntst);

      trainset.Generate(a, p, "Linear", eps);
      testset.Generate(a, p, "Linear", eps);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      int repeat = 10; 

      src::LCurve<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> lcurve(Ns,repeat);
      
      lcurve.Split(trainset, testset, 0., 1.);
      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }
  } 

  TEST_CASE("CLASSIFICATION")
  {
    SUBCASE("BOOT")
    {
      int D, N, Nc;
      D = 1; N = 10; Nc = 2;
      data::classification::Dataset dataset(D, N, Nc);
      std::string type = "Simple";
      dataset.Generate(type);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      int repeat = 10; 
      auto inputs = dataset.inputs_;
      auto labels = dataset.labels_;
      src::LCurve<algo::classification::NMC<>,mlpack::Accuracy,
                  data::N_StratSplit> lcurve(Ns,repeat);
      
      lcurve.Bootstrap(inputs, labels);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) <= 0 );
    }

    SUBCASE("ADD")
    {
      int D, N, Nc;
      D = 1; N = 10; Nc = 2;
      data::classification::Dataset dataset(D, N, Nc);
      std::string type = "Simple";
      dataset.Generate(type);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      int repeat = 10; 
      auto inputs = dataset.inputs_;
      auto labels = dataset.labels_;
      src::LCurve<algo::classification::NMC<>,mlpack::Accuracy,
                  data::N_StratSplit> lcurve(Ns,repeat);
      
      lcurve.Additive(inputs, labels);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) <= 0 );
    }
    SUBCASE("SPLIT")
    {

      std::string type = "Simple";
      int D, Ntrn, Ntst;  D=1; Ntrn=10; Ntst=10;
      data::classification::Dataset trainset(D, Ntrn, 2);
      data::classification::Dataset testset(D, Ntst, 2);

      trainset.Generate(type);
      testset.Generate(type);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5);
      int repeat = 10; 

      src::LCurve<algo::classification::NMC<>,
                  mlpack::Accuracy,
                  data::N_StratSplit> lcurve(Ns,repeat);
      
      lcurve.Split(trainset, testset);
      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) <= 0 );
    }

  }
}

TEST_SUITE("LEARNINGCURVESWITHHPT") {
  TEST_CASE("REGRESSION")
  {
    SUBCASE("BOOT")
    {
      int D, Ntrn;  D=1; Ntrn=20; 
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      data::regression::Dataset dataset(D, Ntrn);

      dataset.Generate(a, p, "Linear", eps);
      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12);
      int repeat = 10; 

      auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10);
      arma::Row<int> hps2 = {true,false};
      src::LCurveHPT<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE>
                                                              lcurve(Ns,repeat);
      
      lcurve.Bootstrap(inputs, labels, hps, hps2);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }
    SUBCASE("BOOT-FIXED")
    {
      int D, Ntrn;  D=1; Ntrn=20; 
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      data::regression::Dataset dataset(D, Ntrn);

      dataset.Generate(a, p, "Linear", eps);
      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12);
      int repeat = 10; 

      auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10);
      bool hps2 = true;

      src::LCurveHPT<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE>
                                                              lcurve(Ns,repeat);
      
      lcurve.Bootstrap(inputs, labels, hps, mlpack::Fixed(hps2));

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }

    SUBCASE("ADD")
    {
      int D, Ntrn;  D=1; Ntrn=100; 
      double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
      data::regression::Dataset dataset(D, Ntrn);

      dataset.Generate(a, p, "Linear", eps);
      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12);
      int repeat = 10; 

      src::LCurveHPT<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> lcurve(Ns,repeat);
      
      auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10);
      arma::Row<int> hps2 = {true,false};
      lcurve.Additive(inputs, labels, hps, hps2);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }

    SUBCASE("SPLIT")
    {
      int D, Ntrn, Ntst;  D=1; Ntrn=100; Ntst=100;
      double a, p, eps; a = 1.0; p = 0.; eps = 0.5;
      data::regression::Dataset trainset(D, Ntrn);
      data::regression::Dataset testset(D, Ntst);

      trainset.Generate(a, p, "Linear", eps);
      testset.Generate(a, p, "Linear", eps);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12);
      int repeat = 10; 

      auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10);
      arma::Row<int> hps2 = {true,false};
      src::LCurveHPT<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> lcurve(Ns,repeat);

      lcurve.Split(trainset, testset, hps, hps2);
      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) > 0 );
    }
  } 

  TEST_CASE("CLASSIFICATION")
  {
    SUBCASE("BOOT")
    {
      int D, N, Nc;
      D = 1; N = 10; Nc = 2;
      data::classification::Dataset dataset(D, N, Nc);
      std::string type = "Simple";
      dataset.Generate(type);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12);
      int repeat = 10; 
      auto inputs = dataset.inputs_;
      auto labels = dataset.labels_;
      src::LCurveHPT<algo::classification::NMC<>,mlpack::Accuracy,
                  data::N_StratSplit> lcurve(Ns,repeat);
      
      auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10);
      lcurve.Bootstrap(inputs, labels,hps);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) <= 0 );
    }

    SUBCASE("ADD")
    {
      int D, N, Nc;
      D = 1; N = 10; Nc = 2;
      data::classification::Dataset dataset(D, N, Nc);
      std::string type = "Simple";
      dataset.Generate(type);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12);
      int repeat = 10; 
      auto inputs = dataset.inputs_;
      auto labels = dataset.labels_;
      src::LCurveHPT<algo::classification::NMC<>,mlpack::Accuracy,
                    data::N_StratSplit> lcurve(Ns,repeat);
      
      auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10);
      lcurve.Additive(inputs, labels,hps);

      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) <= 0 );
    }

    SUBCASE("SPLIT")
    {

      std::string type = "Simple";
      int D, Ntrn, Ntst;  D=1; Ntrn=20; Ntst=10;
      data::classification::Dataset trainset(D, Ntrn, 2);
      data::classification::Dataset testset(D, Ntst, 2);

      trainset.Generate(type);
      testset.Generate(type);

      arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12);
      int repeat = 10; 

      src::LCurveHPT<algo::classification::NMC<>,
                  mlpack::Accuracy,
                  data::N_StratSplit> lcurve(Ns,repeat);
      
      auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10);
      lcurve.Split(trainset, testset,hps);
      CHECK ( arma::mean(arma::mean(
                      (lcurve.test_errors_ - lcurve.train_errors_))) <= 0 );
    }

  }
}

#endif

