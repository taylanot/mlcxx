/**
 * @file test_utils.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_UTILS_H 
#define TEST_UTILS_H

TEST_SUITE("SIMILARITY") {
  
  arma::Row<DTYPE> vec("5 2");
  arma::Row<DTYPE> vec_("4 -10");
  arma::Mat<DTYPE> mat("5 2 ; 4 -10"); 

  arma::Row<DTYPE> curve("1 1 1");
  arma::Row<DTYPE> curve_("0.5 0.5 0.5");
  arma::Mat<DTYPE> curves("1 1 1; 0.5 0.5 0.5");

  size_t same = 1;
  size_t notsame = 0;

  TEST_CASE("COSINE")
  {
    arma::Row<DTYPE> res = utils::cos_sim(mat,vec);
    CHECK ( size_t(res.n_elem) == size_t(mat.n_rows));
    CHECK ( size_t(res(0)) == same );
    CHECK ( size_t(res(1)) == notsame );
  }
  TEST_CASE("CURVE")
  {
    arma::Row<DTYPE> res = utils::curve_sim(curves,curve);
    CHECK ( size_t(res.n_elem) == size_t(curves.n_rows));
    CHECK ( size_t(res(0)) == same );
    CHECK ( res(1) < same );
  }
}
 

#endif
