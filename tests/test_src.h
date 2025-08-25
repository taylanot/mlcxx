/**
 * @file test_src.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_SRC_H 
#define TEST_SRC_H

#include <sys/wait.h>
using CDATASET = data::Dataset<arma::Row<size_t>, DTYPE>;
using RDATASET = data::Dataset<arma::Row<DTYPE>, DTYPE>;
using CMODEL = mlpack::DecisionTree<>;
using RMODEL = mlpack::LinearRegression<>;
using CLOSS = mlpack::Accuracy;
using RLOSS = mlpack::MSE;
using SAMPLE = data::RandomSelect<>;


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



TEST_SUITE("LCurve")
{
  arma::Mat<DTYPE> run_fixed_class()
  {
    CDATASET data(2);
    data.Banana(200);

    auto Ns = arma::regspace<arma::Row<size_t>>(10, 1, 11);

    lcurve::LCurve<CMODEL, CDATASET, SAMPLE, CLOSS, DTYPE> curve(
        data,
        Ns,
        size_t(2),  // reduced for test speed
        true,
        false,
        "temp",
        "cfixed");

    curve.Generate(50);
    return curve.GetResults();
  }

  arma::Mat<DTYPE> run_tuned_class()
  {
    CDATASET data(2);
    data.Banana(200);

    auto Ns = arma::regspace<arma::Row<size_t>>(10, 1, 11);
    auto leafs = arma::regspace<arma::Row<size_t>>(1, 5, 50);

    lcurve::LCurve<CMODEL, CDATASET, SAMPLE, CLOSS, DTYPE> curve(
        data,
        Ns,
        size_t(2),  // reduced for test speed
        true,
        false,
        "temp",
        "ctuned");

    curve.GenerateHpt(0.2, leafs);
    return curve.GetResults();
  }

  arma::Mat<DTYPE> run_fixed_reg()
  {
    RDATASET data(2);
    data.Linear(400);

    auto Ns = arma::regspace<arma::Row<size_t>>(10, 1, 11);

    lcurve::LCurve<RMODEL, RDATASET, SAMPLE, RLOSS, DTYPE> curve(
        data,
        Ns,
        size_t(2),  // reduced for test speed
        true,
        false,
        "temp",
        "rfixed");

    curve.Generate(0);
    return curve.GetResults();
  }

  arma::Mat<DTYPE> run_tuned_reg()
  {
    RDATASET data(1);
    data.Linear(400);

    auto Ns = arma::regspace<arma::Row<size_t>>(10, 1, 11);
    auto ls = arma::regspace<arma::Row<size_t>>(1, 5, 50);

    lcurve::LCurve<RMODEL, RDATASET, SAMPLE, RLOSS, DTYPE> curve(
        data,
        Ns,
        size_t(2),  // reduced for test speed
        true,
        false,
        "temp",
        "rtuned");

    curve.GenerateHpt(0.2, ls);
    return curve.GetResults();
  }

  TEST_CASE("LCurve")
  {
    std::filesystem::create_directories("temp");

    arma::Mat<DTYPE> cfixed = run_fixed_class();
    arma::Mat<DTYPE> ctuned = run_tuned_class();
    CHECK( cfixed.n_rows == 2 );
    CHECK( cfixed.n_cols == 2 );
    CHECK( ctuned.n_rows == 2 );
    CHECK( ctuned.n_cols == 2 ); 
    CHECK( arma::all(arma::mean(ctuned) >= arma::mean(cfixed)) );
    CHECK( std::filesystem::is_regular_file("temp/cfixed"));
    CHECK( std::filesystem::is_regular_file("temp/ctuned"));

    auto load_cfixed = 
      lcurve::LCurve<CMODEL,CDATASET,SAMPLE,CLOSS,DTYPE>::Load("temp/cfixed");
    auto load_ctuned= 
      lcurve::LCurve<CMODEL,CDATASET,SAMPLE,CLOSS,DTYPE>::Load("temp/ctuned");

    CHECK( arma::all(arma::all( load_cfixed->GetResults() == cfixed )) );
    CHECK( arma::all(arma::all( load_ctuned->GetResults() == ctuned )) );

    arma::Mat<DTYPE> rfixed = run_fixed_reg();
    arma::Mat<DTYPE> rtuned = run_tuned_reg();
    CHECK( rfixed.n_rows == 2 );
    CHECK( rfixed.n_cols == 2 );
    CHECK( rtuned.n_rows == 2 );
    CHECK( rtuned.n_cols == 2 );
    CHECK( arma::all(arma::mean(rtuned) <= arma::mean(rfixed)) );

    auto load_rfixed = 
      lcurve::LCurve<RMODEL,CDATASET,SAMPLE,CLOSS,DTYPE>::Load("temp/rfixed");
    auto load_rtuned= 
      lcurve::LCurve<RMODEL,CDATASET,SAMPLE,CLOSS,DTYPE>::Load("temp/rtuned");

    CHECK( arma::all(arma::all( load_rfixed->GetResults() == rfixed )) );
    CHECK( arma::all(arma::all( load_rtuned->GetResults() == rtuned )) );

    std::filesystem::remove_all("temp");
  }
  TEST_CASE("Signal handler exits process") 
  {
    std::filesystem::create_directories("temp");

    CDATASET cdata(2);
    cdata.Banana(200);

    lcurve::LCurve<CMODEL, CDATASET, SAMPLE, CLOSS, DTYPE> ccurve( cdata,
                              {1,2}, size_t(2), false, false, "temp", "ccurve");
     
      // Trigger the signal, which will call _SignalHandler internally
    raise(SIGALRM);

    CHECK( std::filesystem::is_regular_file("temp/ccurve"));

    std::filesystem::remove_all("temp");
  }
}

#endif

