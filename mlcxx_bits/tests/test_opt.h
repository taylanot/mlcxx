/**
 * @file test_opt.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_OPT_H 
#define TEST_OPT_H

TEST_SUITE("FINITE-DIFFERENCE") {

  auto f = [] (const arma::mat& x, const arma::rowvec& p)
  { 
    arma::mat sqr = arma::pow(x,2);
    arma::rowvec y = p*sqr;
    return y;
  };

  auto df = [] (const arma::mat& x)
  { 
    arma::mat sqr = arma::pow(x,2);
    arma::mat dp; dp.eye(x.n_rows, x.n_rows);
    arma::mat dy = dp*sqr;
    return dy;
  };

  TEST_CASE("Jacobian")
  {
    
    int D;
    int N = 10;
    double dp;
    double tol = 1e-6;

    arma::mat x; arma::rowvec y; arma::rowvec p;

    arma::mat J;
    arma::mat J_exact;

    SUBCASE("1D-Central")
    {
      D = 1;
      dp = 0.001;

      x = arma::trans(arma::linspace(1,10,N));

      p.ones(D);
      y = f(x, p);

      J = diff::FD_dfdp(f,x, y, p,dp);
      J_exact = df(x);

      CHECK ( J.n_cols == N );
      CHECK ( J.n_rows == D );
      CHECK ( arma::norm(J-J_exact) <= tol );
    }
    SUBCASE("2D-Central")
    {
      D = 2;
      dp = 0.001;
      arma::mat xi = arma::trans(arma::linspace(1,10,N));

      x = arma::join_cols(xi,xi);

      p.ones(D);
      y = f(x, p);

      J = diff::FD_dfdp(f,x, y, p,dp);
      J_exact = df(x);

      CHECK ( J.n_cols == N );
      CHECK ( J.n_rows == D );
      CHECK ( arma::norm(J-J_exact) <= tol );
    }
    SUBCASE("1D-Backwards")
    {
      D = 1;
      dp = -0.001;

      x = arma::trans(arma::linspace(1,10,N));

      p.ones(D);
      y = f(x, p);

      J = diff::FD_dfdp(f,x, y, p,dp);
      J_exact = df(x);

      CHECK ( J.n_cols == N );
      CHECK ( J.n_rows == D );
      CHECK ( arma::norm(J-J_exact) <= tol );
    }
    SUBCASE("2D-Backwards")
    {
      D = 2;
      dp = -0.001;
      arma::mat xi = arma::trans(arma::linspace(1,10,N));

      x = arma::join_cols(xi,xi);

      p.ones(D);
      y = f(x, p);

      J = diff::FD_dfdp(f,x, y, p,dp);
      J_exact = df(x);

      CHECK ( J.n_cols == N );
      CHECK ( J.n_rows == D );
      CHECK ( arma::norm(J-J_exact) <= tol );
    }
    SUBCASE("BROYDEN_UPDATE-1D")
    {
      D = 1;

      x = arma::trans(arma::linspace(1,10,N));

      arma::rowvec p_old(D,arma::fill::zeros);
      arma::rowvec p(D,arma::fill::ones);
      arma::rowvec y_old = f(x,p_old);
      arma::rowvec y= f(x,p);
      J_exact = df(x);
      arma::mat J_ = diff::J_update(J_exact, p_old, p, y_old, y);
      CHECK ( J_.n_cols == N );
      CHECK ( J_.n_rows == D );
      CHECK ( arma::norm(J_-J_exact) <= tol );
    }
    SUBCASE("BROYDEN_UPDATE-2D")
    {
      D = 2;

      arma::mat xi = arma::trans(arma::linspace(1,10,N));
      x = arma::join_cols(xi,xi);

      arma::rowvec p_old(D,arma::fill::zeros);
      arma::rowvec p(D,arma::fill::ones);
      arma::rowvec y_old = f(x,p_old);
      arma::rowvec y= f(x,p);
      J_exact = df(x);
      arma::mat J_ = diff::J_update(J_exact, p_old, p, y_old, y);
      CHECK ( J_.n_cols == N );
      CHECK ( J_.n_rows == D );
      CHECK ( arma::norm(J_-J_exact) <= tol );
    }

  }
}

TEST_SUITE("LEVENBERG-MARQUARDT") {

  auto f = [] (const arma::mat& x, const arma::rowvec& p)
  { 
    arma::mat sqr = arma::pow(x,2);
    arma::rowvec y = p.t()*sqr;
    return y;
  };

  auto f2 = [] (const arma::mat& x, const arma::rowvec& p)
  { 
    arma::mat sqr = arma::pow(x,2);
    arma::rowvec y = p(0)*sqr+p(1);
    return y;
  };
  auto fe = [] (const arma::mat& x, const arma::rowvec& p)
  { 
    arma::mat exp = arma::exp(x);
    arma::rowvec y = p.t()*exp;
    return y;
  };

  auto fe2 = [] (const arma::mat& x, const arma::rowvec& p)
  { 
    arma::mat exp = arma::exp(x);
    arma::rowvec y = p(0)*exp+p(1);
    return y;
  };
  double tol = 1e-1;

  TEST_CASE("Squared Function")
  {
    
    int D;
    int N = 1000;
    double dp = 0.001;

    arma::mat x; arma::rowvec y; arma::rowvec p;

    arma::mat J;
    arma::mat J_exact;

    SUBCASE("1D-LM")
    {
      D = 1;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p *= 2;
      y = f(x, p);
      p.ones(D);
      opt::LM optimizer;
      optimizer.Optimize( f, x, y, p, dp );
      CHECK ( p(0) - 2. < tol ) ;
    }
    SUBCASE("2D-LM")
    {
      D = 2;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p(1) = 0;
      y = f2(x, p);
      p.ones(D);
      p(1) = 0.1;
      opt::LM optimizer;
      optimizer.Optimize( f2, x, y, p, dp );
      CHECK ( std::abs(p(0) - 1.) <= tol );
      CHECK ( std::abs(p(1)     ) <= tol );
    }
    SUBCASE("1D-Q")
    {
      D = 1;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p *= 2;
      y = f(x, p);
      p.ones(D);
      opt::LM optimizer("Q");
      optimizer.Optimize( f, x, y, p, dp );
      CHECK ( p(0) - 2. < tol ) ;
    }
    SUBCASE("2D-Q")
    {
      D = 2;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p(1) = 0;
      y = f2(x, p);
      p.ones(D);
      p(1) = 0.1;
      opt::LM optimizer("Q");
      optimizer.Optimize( f2, x, y, p, dp );
      CHECK ( std::abs(p(0) - 1.) <= tol );
      CHECK ( std::abs(p(1)     ) <= tol );
    }
    SUBCASE("1D-N")
    {
      D = 1;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p *= 2;
      y = f(x, p);
      p.ones(D);
      opt::LM optimizer("N");
      optimizer.Optimize( f, x, y, p, dp );
      CHECK ( p(0) - 2. < tol ) ;
    }
    SUBCASE("2D-N")
    {
      D = 2;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p(1) = 0;
      y = f2(x, p);
      p.ones(D);
      p(1) = 0.1;
      opt::LM optimizer("N");
      optimizer.Optimize( f2, x, y, p, dp );
      CHECK ( std::abs(p(0) - 1.) <= tol );
      CHECK ( std::abs(p(1)     ) <= tol );
    }
  }

  TEST_CASE("Exponential Function")
  {
    
    int D;
    int N = 1000;
    double dp = 0.001;

    arma::mat x; arma::rowvec y; arma::rowvec p;

    arma::mat J;
    arma::mat J_exact;

    SUBCASE("1D-LM")
    {
      D = 1;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p *= 2;
      y = fe(x, p);
      p.ones(D);
      opt::LM optimizer;
      optimizer.Optimize( fe, x, y, p, dp );
      CHECK ( (p(0) - 2.) < tol );
    }
    SUBCASE("2D-LM")
    {
      D = 2;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p(1) = 0;
      y = fe2(x, p);
      p.ones(D);
      p(1) = 0.1;
      opt::LM optimizer;
      optimizer.Optimize( fe2, x, y, p, dp );
      CHECK ( std::abs(p(0) - 1.) <= tol );
      CHECK ( std::abs(p(1)     ) <= tol );
    }
    SUBCASE("1D-Q")
    {
      D = 1;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p *= 2;
      y = fe(x, p);
      p.ones(D);
      opt::LM optimizer("Q");
      optimizer.Optimize( fe, x, y, p, dp );
      CHECK ( (p(0) - 2.) < tol );
    }
    SUBCASE("2D-Q")
    {
      D = 2;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p(1) = 0;
      y = fe2(x, p);
      p.ones(D);
      p(1) = 0.1;
      opt::LM optimizer("Q");
      optimizer.Optimize( fe2, x, y, p, dp );
      CHECK ( std::abs(p(0) - 1.) <= tol );
      CHECK ( std::abs(p(1)     ) <= tol );
    }
    SUBCASE("1D-N")
    {
      D = 1;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p *= 2;
      y = fe(x, p);
      p.ones(D);
      opt::LM optimizer("N");
      optimizer.Optimize( fe, x, y, p, dp );
      CHECK ( (p(0) - 2.) < tol );
    }
    SUBCASE("2D-N")
    {
      D = 2;
      x = arma::trans(arma::linspace(0,10,N));
      p.ones(D);
      p(1) = 0;
      y = fe2(x, p);
      p.ones(D);
      p(1) = 0.1;
      opt::LM optimizer("N");
      optimizer.Optimize( fe2, x, y, p, dp );
      CHECK ( std::abs(p(0) - 1.) <= tol );
      CHECK ( std::abs(p(1)     ) <= tol );
    }
  }
}

#endif
