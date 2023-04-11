/**
 * @file test_stats.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_STATS_H 
#define TEST_STATS_H

TEST_SUITE("HYPOTHESIS") {
  arma::rowvec b = {1,2,3,4};
  TEST_CASE("CramerVonMisses-2Sample_Approximaate")
  {
    CHECK ( stats::cramervonmisses_2samp(b,b,"asymptotic") == 1. );
  }
}

#endif
