/**
 * @file test_stats.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_STATS_H 
#define TEST_STATS_H

TEST_SUITE("HYPOTHESIS") {
  using Vec = arma::Row<DTYPE>;
  DTYPE alpha = 0.5;
  TEST_CASE("CramerVonMisses-2Sample_Approximaate")
  {
    Vec v = {1,2,3,4};
    CHECK ( stats::cramervonmisses_2samp(v,v,"asymptotic") == 1. );
  }
  TEST_CASE("qtest")
  {
    Vec x = arma::randu<arma::Row<DTYPE>>(10);
    Vec y = arma::randu<arma::Row<DTYPE>>(10, arma::distr_param(1,2));
    CHECK ( 1.-stats::qtest<Vec,double>(0,x,y) < alpha/2 );
  }
}

TEST_SUITE("HIGHMOMENTS") {
  arma::Mat<DTYPE> data = {1,2,3,4,5};
  arma::Mat<DTYPE> data1 = {{1,2,3,4,5},{2,3,4,5,6}};

  TEST_CASE("SKEW")
  {
    CHECK( stats::Skew(data).eval()(0) == DTYPE(0) );
    CHECK( stats::Skew(data1).eval()(1) == DTYPE(0) );
  }

  TEST_CASE("KURTOSIS")
  {
    CHECK( stats::Kurt(data).eval()(0) == DTYPE(-1.3) );
    CHECK( stats::Kurt(data1).eval()(1) == DTYPE(-1.3) );
  }

}
TEST_SUITE("MEANESTIMATION") {

  arma::Mat<DTYPE> data = arma::randn< arma::Mat<DTYPE>>(1,1000);
  double tol = 1.e-6;

  TEST_CASE("LEE")
  {
    CHECK( std::abs(stats::lee(data).eval()(0,0)-
                              arma::mean(data,1).eval()(0)) > tol );
  }

  TEST_CASE("CATONI")
  {
    CHECK( 
        std::abs(stats::catoni(data).eval()(0,0)-
                              arma::mean(data,1).eval()(0)) > tol );
  }

  TEST_CASE("TRIMMED")
  {
    CHECK( 
        std::abs(stats::tmean(data,10).eval()(0,0)-
                              arma::mean(data,1).eval()(0)) > tol );
  }

  TEST_CASE("MEDIANOFMEANS")
  {
    CHECK( 
        std::abs(stats::mediomean(data,10).eval()(0)-
                              arma::mean(data,1).eval()(0)) > tol );
  }

}
#endif
