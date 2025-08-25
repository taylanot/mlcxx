/**
 * @file test_algo.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_ALGO_H 
#define TEST_ALGO_H

template<typename T=DTYPE>
class IdentityKernel
{
public:
  IdentityKernel() = default;

  DTYPE Evaluate (const arma::Mat<T>& X1, const arma::Mat<T>& X2) const
  {
    if (arma::approx_equal(X1, X2, "absdiff", 1e-12))
      return T(1.);
    else
      return T(0.);
  }
};

// Dummy FUNC: Identity features
template<typename T=DTYPE>
struct IdentityFeatures
{
  size_t M_;
  IdentityFeatures(size_t M = 1) : M_(M) {}

  arma::Mat<T> Predict(const arma::Mat<T>& X) const
  {
    // Just return first M_ rows of X (or X if fewer rows)
    return X.rows(0, std::min(M_, (size_t)X.n_rows) - 1);
  }

  arma::Row<T> Mean(const arma::Mat<T>& X) const
  {
    // Zero mean function
    return arma::Row<T>(X.n_cols, arma::fill::zeros);
  }

  size_t GetM() const { return M_; }

  // --- cereal serialization ---
  template<class Archive>
  void serialize(Archive& ar) { ar(M_); }
};

// Networks
using RNetworkType = mlpack::FFN<mlpack::MeanSquaredError>;
using CNetworkType = mlpack::FFN<mlpack::CrossEntropyError>;

// Optimizers 
using OPT = ens::Adam;

// LOSES
using ACC = mlpack::Accuracy;
using MSE = mlpack::MSE;

// ANN models
using RANN = algo::ANN<RNetworkType,OPT,MSE>;
using CANN = algo::ANN<CNetworkType,OPT,ACC>;
// Regression models
using GPR = algo::regression::GaussianProcess<mlpack::GaussianKernel>;
using KR = algo::regression::KernelRidge<mlpack::GaussianKernel>;
using KWR = algo::regression::Kernel<IdentityKernel<>>;
using SPKR = algo::regression::SemiParamKernelRidge<mlpack::GaussianKernel,IdentityFeatures<>>;
// Classification models
using LDC = algo::classification::LDC<>;
using QDC = algo::classification::QDC<>;
using NNC = algo::classification::NNC<>;
using NMC = algo::classification::NMC<>;
using OvA = algo::classification::OnevAll<mlpack::LogisticRegression<>>;
using LREG = algo::classification::LogisticRegression<>;
using SVM = algo::classification::SVM<mlpack::GaussianKernel>;




// Utility: check save/load consistency using fully qualified std::filesystem
template<class MODEL, class LabelType, class METRIC>
void CheckModelPersistence( MODEL& model,
                            const arma::mat& inputs,
                            const LabelType& labels, METRIC metric)
{
  std::filesystem::create_directories("temp");
  std::filesystem::path tmpFile = "temp/model.bin";
  
  algo::save<MODEL>(tmpFile,model);
                                              
  auto loaded = algo::load<MODEL>(tmpFile);

  auto res = metric.Evaluate(model,inputs,labels);
  auto res2 = metric.Evaluate(*loaded,inputs,labels);

  CHECK( res == doctest::Approx(res2) );

  std::filesystem::remove_all("temp");
}

template<typename MODEL, typename Metric=ACC>
void TestLinearClassifier()
{
  arma::Mat<DTYPE> inputs(2, 6);
  inputs.col(0) = {0.0, 0.0};
  inputs.col(1) = {0.1, 0.1};
  inputs.col(2) = {0.2, 0.0};
  inputs.col(3) = {1.0, 1.0};
  inputs.col(4) = {1.1, 1.0};
  inputs.col(5) = {0.9, 1.2};

  arma::Row<size_t> labels = {0, 0, 0, 1, 1, 1};

  MODEL model(inputs, labels,2);
  Metric metric;

  DTYPE acc = metric.Evaluate(model, inputs, labels);

  arma::Row<size_t> preds;
  model.Classify(inputs, preds);
  CHECK(preds.n_elem == labels.n_elem);
  CHECK(acc == doctest::Approx(1.0));

  CheckModelPersistence(model, inputs, labels,metric);
}

template<typename MODEL, typename Metric=ACC>
void TestNonLinearClassifier()
{
  arma::Mat<DTYPE> inputs(2, 4);
  inputs.col(0) = {0.0, 0.0};
  inputs.col(1) = {0.0, 1.0};
  inputs.col(2) = {1.0, 0.0};
  inputs.col(3) = {1.0, 1.0};

  arma::Row<size_t> labels = {0, 1, 1, 0};

  MODEL model(inputs, labels,2);
  Metric metric;

  DTYPE acc = metric.Evaluate(model, inputs, labels);

  arma::Row<size_t> preds;
  model.Classify(inputs, preds);
  CHECK(preds.n_elem == labels.n_elem);
  CHECK(acc == doctest::Approx(1.0));

  CheckModelPersistence(model, inputs, labels, metric);
}

template<typename MODEL, typename Metric=ACC>
void TestOneClassProblem()
{
  arma::mat inputs(2, 5, arma::fill::randu);
  arma::Row<size_t> labels = {0, 0, 0, 0, 0};

  MODEL model(inputs, labels,1);

  arma::Row<size_t> preds;
  model.Classify(inputs, preds);
  CHECK(preds.n_elem == labels.n_elem);
  CHECK(arma::all(preds == 0)); 

}

template<typename MODEL, typename Metric=ACC>
void TestOneSampleProblem()
{
  arma::mat inputs(2, 1, arma::fill::randu);
  arma::Row<size_t> labels = {0};

  MODEL model(inputs, labels,1);

  arma::Row<size_t> preds;
  model.Classify(inputs, preds);
  CHECK(preds.n_elem == labels.n_elem);
  CHECK(preds[0] == labels[0]); 

}

template<typename MODEL, typename Metric=MSE>
void TestMemorize()
{
  // Training data
  arma::Row<DTYPE> x = {1.0, 2.0, 3.0};
  arma::Mat<DTYPE> inputs(1, x.n_elem, arma::fill::zeros);
  inputs.row(0) = x;
  arma::Row<DTYPE> labels = {10.0, 20.0, 30.0};

  MODEL model(inputs, labels);
  Metric metric;

  // Predict on training points
  arma::Row<DTYPE> preds;
  model.Predict(inputs, preds);

  CHECK(preds.n_elem == labels.n_elem);
  for (size_t i = 0; i < labels.n_elem; ++i)
  {
    CHECK(preds(i) == doctest::Approx(labels(i)));
  }

  // MSE should be zero
  DTYPE mse = metric.Evaluate(model, inputs, labels);
  CHECK(mse == doctest::Approx(0.0).epsilon(1e-6));

  // Persistence check
  CheckModelPersistence(model, inputs, labels, metric);
}

// Nonlinear regression test: quadratic relation
template<typename MODEL, typename Metric=MSE>
void TestNonLinearRegression()
{
  arma::Row<DTYPE> x = arma::linspace<arma::rowvec>(-2.0, 2.0, 50);
  arma::Mat<DTYPE> inputs(1, x.n_elem);
  inputs.row(0) = x;

  // True function: y = x^2
  arma::Row<DTYPE> labels = arma::square(x);

  MODEL model(inputs, labels);
  Metric metric;

  arma::Row<DTYPE> preds;
  model.Predict(inputs, preds);

  CHECK(preds.n_elem == labels.n_elem);

  DTYPE mse = metric.Evaluate(model,inputs, labels);
  CHECK(mse == doctest::Approx(0.0).epsilon(1e-6));

  CheckModelPersistence(model, inputs, labels, metric);
}

TEST_SUITE("CLASSIFIERS") 
{

  TEST_CASE("LDC")
  {
    TestOneClassProblem<LDC>();
    TestOneSampleProblem<LDC>();
    TestLinearClassifier<LDC>();
  }

  TEST_CASE("NMC")
  {
    TestOneClassProblem<NMC>();
    TestOneSampleProblem<NMC>();
    TestLinearClassifier<NMC>();
  }
  TEST_CASE("NNC")
  {
    TestOneClassProblem<NNC>();
    TestOneSampleProblem<NNC>();
    TestNonLinearClassifier<NNC>();
  }

  TEST_CASE("QDC")
  {
    TestOneClassProblem<QDC>();
    TestOneSampleProblem<QDC>();
    TestNonLinearClassifier<QDC>();
  }

  TEST_CASE("OvA")
  {
    TestOneClassProblem<OvA>();
    TestOneSampleProblem<OvA>();
    TestLinearClassifier<OvA>();
  }

  TEST_CASE("LREG")
  {
    TestOneClassProblem<LREG>();
    TestOneSampleProblem<LREG>();
    TestLinearClassifier<LREG>();
  }

  TEST_CASE("SVM")
  {
    TestOneClassProblem<SVM>();
    TestOneSampleProblem<SVM>();
    TestNonLinearClassifier<SVM>();
  }

} 

TEST_SUITE("REGRESSORS") 
{
  TEST_CASE("KR")
  {
    TestNonLinearRegression<KR>();
  }

  TEST_CASE("KWR")
  {
    TestMemorize<KWR>();
  }

  TEST_CASE("GPR")
  {
    TestNonLinearRegression<GPR>();

    // --- Training data: y = x^2 ---
    arma::Row<DTYPE> x = arma::linspace<arma::rowvec>(-2.0, 2.0, 15);
    arma::Mat<DTYPE> inputs(1, x.n_elem); 
    inputs.row(0) = x;
    arma::Row<DTYPE> labels = arma::square(x);

    // Construct GP
    GPR gp(inputs, labels);

    SUBCASE("PredictVariance")
    {
      arma::Mat<DTYPE> var;
      gp.PredictVariance(inputs, var);

      CHECK(var.n_rows == inputs.n_cols);
      CHECK(var.n_cols == inputs.n_cols);

      // Must be symmetric
      CHECK(arma::approx_equal(var, var.t(), "absdiff", 1e-12));

      // Variance at training points should be ~0
      for(size_t i=0; i<var.n_rows; i++)
        CHECK(var(i,i) == doctest::Approx(0.0).epsilon(1e-6));
    }

    SUBCASE("ComputeError")
    {
      DTYPE err = gp.ComputeError(inputs, labels);
      CHECK(err == doctest::Approx(0.0).epsilon(1e-6));
    }

    SUBCASE("LogLikelihood")
    {
      DTYPE ll = gp.LogLikelihood(inputs, labels);
      CHECK(std::isfinite(ll));
      CHECK(ll < 0.0);
    }

    SUBCASE("SamplePosterior")
    {
      arma::Mat<DTYPE> samples;
      gp.SamplePosterior(500, inputs, samples);

      arma::Row<DTYPE> pred_mean;
      gp.Predict(inputs, pred_mean);

      arma::Row<DTYPE> sample_mean = arma::mean(samples, 0);

      for(size_t i=0; i<pred_mean.n_elem; i++)
        CHECK(sample_mean(i) == doctest::Approx(pred_mean(i)).epsilon(0.2));
    }

    SUBCASE("SamplePrior")
    {
      GPR gp_prior; // default GP with no training

      arma::Mat<DTYPE> samples;
      gp_prior.SamplePrior(500, inputs, samples);

      arma::Row<DTYPE> mean = arma::mean(samples, 0);
      for(size_t i=0; i<mean.n_elem; i++)
        CHECK(mean(i) == doctest::Approx(0.0).epsilon(0.2));
    }
  }

  TEST_CASE("SPKR")
  {
    // --- Synthetic quadratic data ---
    arma::Row<DTYPE> x = arma::linspace<arma::rowvec>(-2.0, 2.0, 25);
    arma::Mat<DTYPE> inputs(1, x.n_elem);
    inputs.row(0) = x;
    arma::Row<DTYPE> labels = arma::square(x);

    SUBCASE("Size and Error")
    {
      SPKR model( inputs, labels, 0., size_t(1), DTYPE(1.0));

      arma::Row<DTYPE> preds;
      model.Predict(inputs, preds);

      CHECK(preds.n_elem == labels.n_elem);

      DTYPE err = model.ComputeError(inputs, labels);
      CHECK(err == doctest::Approx(0.0).epsilon(1e-6));
    }
  }
}


TEST_SUITE("DIMRED") 
{
    TEST_CASE("UFPCA")
    {
      SUBCASE("wihout mean")
      {
        arma::Row<DTYPE> x = arma::linspace<arma::Row<DTYPE>>(0, 1, 5);
        arma::Mat<DTYPE> inputs(1, 5);
        inputs.row(0) = x;

        // create labels: 2 functions, sine and cosine
        arma::Mat<DTYPE> labels(2, 5);
        labels.row(0) = arma::sin(2 * arma::datum::pi * x);
        labels.row(1) = arma::cos(2 * arma::datum::pi * x);

        auto [eigenvalues, eigenfunctions] = algo::dimred::ufpca(inputs,labels);

        CHECK(eigenvalues.n_elem == 5);
        CHECK(eigenfunctions.n_rows == 5);
        CHECK(eigenfunctions.n_cols == 5);

        // eigenvalues must be nonnegative and sorted descending
        CHECK(arma::all(eigenvalues >= 0));
        CHECK(eigenvalues.is_sorted("descend"));
      }

      SUBCASE("ufpca with mean add")
      {
        arma::Row<DTYPE> x = arma::linspace<arma::Row<double>>(0, 1, 5);
        arma::Mat<DTYPE> inputs(1, 5);
        inputs.row(0) = x;

        arma::Mat<DTYPE> labels(2, 5, arma::fill::randu);

        auto [eig1, funcs1] = algo::dimred::ufpca(inputs, labels, false);
        auto [eig2, funcs2] = algo::dimred::ufpca(inputs, labels, true);

        // mean_add should shift eigenfunctions
        CHECK_FALSE(arma::approx_equal(funcs1, funcs2, "absdiff", 1e-12));
      }

      SUBCASE("ufpca with percentage of variance (ppc)")
      {
        arma::Row<DTYPE> x = arma::linspace<arma::Row<DTYPE>>(0, 1, 10);
        arma::Mat<DTYPE> inputs(1, 10);
        inputs.row(0) = x;

        arma::Mat<double> labels(3, 10, arma::fill::randu);

        auto [eigvals, eigfuncs] = algo::dimred::ufpca(inputs, labels, 0.9);

        CHECK(eigvals.n_elem <= 10);
        CHECK(eigfuncs.n_rows == eigvals.n_elem);
        CHECK(eigfuncs.n_cols == 10);
      }

      SUBCASE("ufpca with fixed number of components (npc)")
      {
        arma::Row<DTYPE> x = arma::linspace<arma::Row<DTYPE>>(0, 1, 8);
        arma::Mat<DTYPE> inputs(1, 8);
        inputs.row(0) = x;

        arma::Mat<DTYPE> labels(4, 8, arma::fill::randu);

        auto [eigvals, eigfuncs] = algo::dimred::ufpca(inputs,labels,size_t(3));

        CHECK(eigvals.n_elem == 3);
        CHECK(eigfuncs.n_rows == 3);
        CHECK(eigfuncs.n_cols == 8);
      }
    }
}

TEST_SUITE("NN")
{
  TEST_CASE("Regression")
  {
    // Synthetic regression dataset: y = sum(x)
    arma::Mat<DTYPE> X = arma::randu<arma::mat>(3, 100);    // 3 features, 100 samples
    arma::Row<DTYPE> y = arma::sum(X, 0);                // labels

    // Build regression network
    RNetworkType rnetwork;
    rnetwork.Add<mlpack::Linear>(5);
    rnetwork.Add<mlpack::ReLU>();
    rnetwork.Add<mlpack::Linear>(5);
    rnetwork.Add<mlpack::ReLU>();
    rnetwork.Add<mlpack::Linear>(1);

    RANN ann(X, y, rnetwork);
    MSE metric;
    
    arma::Mat<DTYPE> preds;
    ann.Predict(X, preds);

    CHECK(preds.n_elem == y.n_elem);

    DTYPE mse = metric.Evaluate(ann, X, y);

    CHECK( mse == doctest::Approx(0).epsilon(1e-3) );

    // Test persistence
    CheckModelPersistence(ann, X, y, metric);
  }

  TEST_CASE("Classification")
  {
    arma::Mat<DTYPE> X(2, 4);
    X.col(0) = {0.0, 0.0};
    X.col(1) = {0.0, 1.0};
    X.col(2) = {1.0, 0.0};
    X.col(3) = {1.0, 1.0};

    arma::Row<size_t> y = {0, 1, 1, 0};

    // Build classification network
    CNetworkType cnetwork;
    cnetwork.Add<mlpack::Linear>(5);
    cnetwork.Add<mlpack::ReLU>();
    cnetwork.Add<mlpack::Linear>(5);
    cnetwork.Add<mlpack::ReLU>();
    cnetwork.Add<mlpack::Linear>(2);
    cnetwork.Add<mlpack::Softmax>();

    CANN ann(X, y, cnetwork);

    arma::Row<size_t> preds;
    ann.Classify(X, preds);

    CHECK(preds.n_elem == y.n_elem);

    ACC acc;
    DTYPE res = acc.Evaluate(ann,X,y);

    CHECK( res == doctest::Approx(1.).epsilon(1e-2) );  // Should learn simple boundary

    // Test persistence
    ACC metric;
    CheckModelPersistence(ann, X, y, metric);
  }
}





#endif
