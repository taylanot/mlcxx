/**
 * @file test_algo.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_ALGO_H 
#define TEST_ALGO_H

TEST_SUITE("NMC") {
  TEST_CASE("Check")
  {
    int D, N, Nc; //double tol = 1e-2;
    std::string type;

    SUBCASE("1D-Simple")
    {
      D = 1; N = 10000; Nc = 2; type = "Simple";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("2D-Simple")
    {
      D = 2; N = 10000; Nc = 2; type = "Simple";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 10000; Nc = 2; type = "Hard";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 10000; Nc = 2; type = "Hard";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
  }
}

TEST_SUITE("SMPKR") {
  TEST_CASE("Check")
  {
    CHECK( 1 == 1 );
  }
}

#endif
