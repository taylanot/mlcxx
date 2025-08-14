/**
 * @file test_src.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_SRC_H 
#define TEST_SRC_H

//-----------------------------------------------------------------------------
// Dummy classification model
//-----------------------------------------------------------------------------
class DummyClassifier
{
public:
  void Classify(const arma::Mat<DTYPE>& data,
                arma::Row<size_t>& predictions,
                arma::Mat<DTYPE>& probabilities)
  {
    size_t n = data.n_cols;
    probabilities.set_size(2, n);
    for (size_t i = 0; i < n; ++i)
    {
      probabilities(0, i) = 0.8; // class 0
      probabilities(1, i) = 0.2; // class 1
    }
    predictions.set_size(n);
    predictions.fill(0); // always predict class 0
  }
  void Classify(const arma::Mat<DTYPE>& data,
                arma::Row<size_t>& predictions)
  {
    predictions.set_size(data.n_cols);
    predictions.fill(0); // always predict class 0
  }
};

//-----------------------------------------------------------------------------
// Dummy regression model
//-----------------------------------------------------------------------------
class DummyRegressor
{
public:
  void Predict(const arma::Mat<DTYPE>& data,
               arma::Row<DTYPE>& predictions)
  {
    predictions.set_size(data.n_cols);
    predictions.fill(1.0); // always predict 3.5
  }
};


TEST_SUITE("METRICS")
{
  // A random small dataset
  static arma::Mat<DTYPE> X = arma::randu<arma::Mat<DTYPE>>(2, 3);
  static arma::Row<size_t> y_class = {0, 1, 0};
  static arma::Row<DTYPE> y_reg = {2.0, 3.0, 4.0};


  TEST_CASE("CrossEntropy metric test")
  {
    DummyClassifier clf;
    DTYPE cross_entropy = metrics::CrossEntropy::Evaluate(clf, X, y_class);
    CHECK(cross_entropy == doctest::Approx(0.685242));
  }

  TEST_CASE("BrierLoss metric test")
  {
    DummyClassifier clf;
    DTYPE brier = metrics::BrierLoss::Evaluate(clf, X, y_class);
    DTYPE expected = ( (0.2 - 0)*(0.2 - 0) +
                       (0.2 - 1)*(0.2 - 1) +
                       (0.2 - 0)*(0.2 - 0) ) / 3.0;
    CHECK(brier == doctest::Approx(expected));
  }

  TEST_CASE("ErrorRate metric test")
  {
    DummyClassifier clf;
    DTYPE error_rate = metrics::ErrorRate::Evaluate(clf, X, y_class);
    DTYPE expected = 1.0/3.0;
    CHECK(error_rate == doctest::Approx(expected));
  }

  TEST_CASE("MSEClass metric test")
  {
    DummyClassifier clf;
    DTYPE mse = metrics::MSEClass::Evaluate(clf, X, y_class);
    DTYPE expected = ( std::pow(1,2) ) / 3.0;
    CHECK(mse == doctest::Approx(expected));
  }

  TEST_CASE("AUC metric test")
  {
    DummyClassifier clf;
    DTYPE auc = metrics::AUC::Evaluate(clf, X, y_class);
    CHECK(auc == doctest::Approx(0.5));
  }
}

/* bool is_decr(const arma::Row<DTYPE>& vec) */ 
/* { */
/*   return vec(0) > vec(vec.n_elem-1); */
/* } */

/* TEST_SUITE("LEARNINGCURVES") { */
/*   TEST_CASE("REGRESSION") */
/*   { */
/*     SUBCASE("BOOT") */
/*     { */
/*       int D, Ntrn;  D=1; Ntrn=100; */ 
/*       DTYPE a, p, eps; a = 1.0; p = 0.; eps = 0.1; */
/*       data::regression::Dataset dataset(D, Ntrn); */

/*       dataset.Generate(a, p, "Linear", eps); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(2,50,100); */
/*       int repeat = 100; */ 

/*       src::LCurve<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> */
/*                                                               lcurve(Ns,repeat); */
      
/*       lcurve.Bootstrap(dataset, 0., 1.); */

/*       CHECK (is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("RANDOM") */
/*     { */
/*       int D, Ntrn;  D=1; Ntrn=100; */ 
/*       DTYPE a, p, eps; a = 1.0; p = 0.; eps = 0.1; */
/*       data::regression::Dataset dataset(D, Ntrn); */

/*       dataset.Generate(a, p, "Linear", eps); */
/*       auto inputs = dataset.inputs_; */
/*       auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(2,50,100); */
/*       int repeat = 100; */ 

/*       src::LCurve<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> */
/*                                                               lcurve(Ns,repeat); */
      
/*       lcurve.RandomSet(inputs, labels, 0., 1.); */

/*       CHECK (is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("ADD") */
/*     { */
/*       int D, Ntrn;  D=1; Ntrn=100; */ 
/*       DTYPE a, p, eps; a = 1.0; p = 0.; eps = 0.1; */
/*       data::regression::Dataset dataset(D, Ntrn); */

/*       dataset.Generate(a, p, "Linear", eps); */
/*       auto inputs = dataset.inputs_; */
/*       auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(2,50,100); */
/*       int repeat = 100; */ 

/*       src::LCurve<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> lcurve(Ns,repeat); */
      
/*       lcurve.Additive(inputs, labels, 0., 1.); */

/*       CHECK (is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("SPLIT") */
/*     { */
/*       int D, Ntrn, Ntst;  D=1; Ntrn=100; Ntst=100; */
/*       DTYPE a, p, eps; a = 1.0; p = 0.; eps = 0.1; */
/*       data::regression::Dataset trainset(D, Ntrn); */
/*       data::regression::Dataset testset(D, Ntst); */

/*       trainset.Generate(a, p, "Linear", eps); */
/*       testset.Generate(a, p, "Linear", eps); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(2,50,100); */
/*       int repeat = 100; */ 

/*       src::LCurve<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> lcurve(Ns,repeat); */
      
/*       lcurve.Split(trainset, testset, 0., 1.); */
/*       CHECK (is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */
/*   } */ 

/*   TEST_CASE("CLASSIFICATION") */
/*   { */
/*     SUBCASE("BOOT") */
/*     { */
/*       int D, N, Nc; */
/*       D = 1; N = 10; Nc = 2; */
/*       data::classification::Dataset dataset(D, N, Nc); */
/*       std::string type = "Simple"; */
/*       dataset.Generate(type); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5); */
/*       int repeat = 100; */ 
/*       src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> */
/*         lcurve(Ns,repeat); */
      
/*       lcurve.Bootstrap(dataset, Nc); */

/*       CHECK (!is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */
/*     SUBCASE("RANDOM") */
/*     { */
/*       int D, N, Nc; */
/*       D = 1; N = 10; Nc = 2; */
/*       data::classification::Dataset dataset(D, N, Nc); */
/*       std::string type = "Simple"; */
/*       dataset.Generate(type); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5); */
/*       int repeat = 100; */ 
/*       auto inputs = dataset.inputs_; */
/*       auto labels = dataset.labels_; */
/*       src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat); */
      
/*       lcurve.RandomSet<data::N_StratSplit>(inputs, labels, Nc); */

/*       CHECK (!is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("ADD") */
/*     { */
/*       int D, N, Nc; */
/*       D = 1; N = 10; Nc = 2; */
/*       data::classification::Dataset dataset(D, N, Nc); */
/*       std::string type = "Simple"; */
/*       dataset.Generate(type); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5); */
/*       int repeat = 100; */ 
/*       auto inputs = dataset.inputs_; */
/*       auto labels = dataset.labels_; */
/*       src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat); */
      
/*       lcurve.Additive<data::N_StratSplit>(inputs, labels, Nc); */

/*       CHECK (!is_decr(arma::mean(lcurve.GetResults()))); */

/*     } */
/*     SUBCASE("SPLIT") */
/*     { */

/*       std::string type = "Simple"; */
/*       int D, Ntrn, Ntst, Nc;  D=1; Ntrn=10; Ntst=10; Nc = 2; */
/*       data::classification::Dataset trainset(D, Ntrn, Nc); */
/*       data::classification::Dataset testset(D, Ntst, Nc); */

/*       trainset.Generate(type); */
/*       testset.Generate(type); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5); */
/*       int repeat = 100; */ 

/*       src::LCurve<algo::classification::NMC<>, */
/*                   mlpack::Accuracy> lcurve(Ns,repeat); */
      
/*       lcurve.Split<data::N_StratSplit>(trainset, testset, Nc); */

/*       CHECK (!is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*   } */
/* } */

/* TEST_SUITE("LEARNINGCURVESWITHHPT") { */
/*   TEST_CASE("REGRESSION") */
/*   { */
/*     SUBCASE("BOOT") */
/*     { */
/*       int D, Ntrn;  D=1; Ntrn=100; */ 
/*       DTYPE a, p, eps; a = 1.0; p = 0.; eps = 1.0; */
/*       data::regression::Dataset dataset(D, Ntrn); */

/*       dataset.Generate(a, p, "Linear", eps); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(10,50,100); */
/*       int repeat = 100; */ 

/*       auto hps = arma::logspace<arma::Row<DTYPE>>(-6,10,10); */
/*       bool hps2 = true; */

/*       src::LCurveHPT<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> */
/*                                                               lcurve(Ns,repeat); */
      
/*       lcurve.Bootstrap(dataset, hps, mlpack::Fixed(hps2)); */

/*       CHECK (is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("RANDOM") */
/*     { */
/*       int D, Ntrn;  D=1; Ntrn=100; */ 
/*       DTYPE a, p, eps; a = 1.0; p = 0.; eps = 1.0; */
/*       data::regression::Dataset dataset(D, Ntrn); */

/*       dataset.Generate(a, p, "Linear", eps); */
/*       auto inputs = dataset.inputs_; */
/*       auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(10,50,100); */
/*       int repeat = 100; */ 

/*       auto hps = arma::logspace<arma::Row<DTYPE>>(-6,10,10); */
/*       bool hps2 = true; */

/*       src::LCurveHPT<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> */
/*                                                               lcurve(Ns,repeat); */
      
/*       lcurve.RandomSet(inputs, labels, hps, mlpack::Fixed(hps2)); */

/*       CHECK (is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("ADD") */
/*     { */
/*       int D, Ntrn;  D=1; Ntrn=100; */ 
/*       DTYPE a, p, eps; a = 1.0; p = 0.; eps = 1.0; */
/*       data::regression::Dataset dataset(D, Ntrn); */

/*       dataset.Generate(a, p, "Linear", eps); */
/*       auto inputs = dataset.inputs_; */
/*       auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(10,50,100); */
/*       int repeat = 100; */ 

/*       src::LCurveHPT<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> */
/*         lcurve(Ns,repeat); */
      
/*       auto hps = arma::logspace<arma::Row<DTYPE>>(-6,10,10); */
/*       lcurve.Additive(inputs, labels, hps, mlpack::Fixed(true)); */

/*       CHECK (is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("SPLIT") */
/*     { */
/*       int D, Ntrn, Ntst;  D=1; Ntrn=100; Ntst=100; */
/*       DTYPE a, p, eps; a = 1.0; p = 0.; eps = 1.0; */
/*       data::regression::Dataset trainset(D, Ntrn); */
/*       data::regression::Dataset testset(D, Ntst); */

/*       trainset.Generate(a, p, "Linear", eps); */
/*       testset.Generate(a, p, "Linear", eps); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(10,50,100); */
/*       int repeat = 100; */ 

/*       auto hps = arma::logspace<arma::Row<DTYPE>>(-6,10,10); */
/*       src::LCurveHPT<mlpack::LinearRegression<arma::Mat<DTYPE>>,mlpack::MSE> */ 
/*         lcurve(Ns,repeat); */

/*       lcurve.Split(trainset, testset, hps, mlpack::Fixed(true)); */

/*       CHECK (is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */
/*   } */ 

/*   TEST_CASE("CLASSIFICATION") */
/*   { */
/*     using MODEL = algo::classification::NMC<>; */
/*     SUBCASE("BOOT") */
/*     { */
/*       int D, N, Nc; */
/*       D = 1; N = 100; Nc = 2; */
/*       data::classification::Dataset dataset(D, N, Nc); */
/*       std::string type = "Simple"; */
/*       dataset.Generate(type); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12); */
/*       int repeat = 10; */ 
/*       src::LCurveHPT<MODEL,mlpack::Accuracy> lcurve(Ns,repeat); */
      
/*       auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10); */
/*       lcurve.Bootstrap(dataset,hps); */

/*       CHECK (!is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */
/*     SUBCASE("RANDOM") */
/*     { */
/*       int D, N, Nc; */
/*       D = 1; N = 100; Nc = 2; */
/*       data::classification::Dataset dataset(D, N, Nc); */
/*       std::string type = "Simple"; */
/*       dataset.Generate(type); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12); */
/*       int repeat = 100; */ 
/*       auto inputs = dataset.inputs_; */
/*       auto labels = dataset.labels_; */
/*       src::LCurveHPT<MODEL, */
/*                      mlpack::Accuracy> lcurve(Ns,repeat); */
      
/*       auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10); */
/*       lcurve.RandomSet<data::N_StratSplit>(inputs,labels,hps); */

/*       CHECK (!is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("ADD") */
/*     { */
/*       int D, N, Nc; */
/*       D = 1; N = 100; Nc = 2; */
/*       data::classification::Dataset dataset(D, N, Nc); */
/*       std::string type = "Simple"; */
/*       dataset.Generate(type); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12); */
/*       int repeat = 100; */ 
/*       auto inputs = dataset.inputs_; */
/*       auto labels = dataset.labels_; */
/*       src::LCurveHPT<MODEL, */
/*                      mlpack::Accuracy> lcurve(Ns,repeat); */
      
/*       auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10); */
/*       lcurve.Additive<data::N_StratSplit>(inputs,labels,hps); */

/*       CHECK (!is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*     SUBCASE("SPLIT") */
/*     { */

/*       std::string type = "Simple"; */
/*       int D, Ntrn, Ntst, Nc;  D=1; Ntrn=20; Ntst=10; Nc=2; */
/*       data::classification::Dataset trainset(D, Ntrn, Nc); */
/*       data::classification::Dataset testset(D, Ntst, Nc); */

/*       trainset.Generate(type); */
/*       testset.Generate(type); */

/*       arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12); */
/*       int repeat = 100; */ 

/*       src::LCurveHPT<MODEL, */
/*                   mlpack::Accuracy> lcurve(Ns,repeat); */
      
/*       auto hps = arma::linspace<arma::Row<DTYPE>>(0.01,1,10); */
/*       lcurve.Split<data::N_StratSplit>(trainset,testset,hps); */

/*       CHECK (!is_decr(arma::mean(lcurve.GetResults()))); */
/*     } */

/*   } */
/* } */

#endif

