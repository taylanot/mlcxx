/**
 *
 * @file realworld.h
 * @author Ozgur Taylan Turan
 *
 * Generation of real world dataset learning curves
 */
namespace  experiments {
namespace lci {

void real_class1 ( )
{
    utils::data::classification::Dataset dataset;
    dataset.Load("datasets/winequality-white.csv");

    utils::data::classification::Dataset trainset, testset;
    utils::data::StratifiedSplit(dataset, trainset, testset, 0.2);

    arma::irowvec Ns = arma::regspace<arma::irowvec>(2,10,3900);

    src::classification::LCurve<algo::classification::NNC,
                                  mlpack::Accuracy> lcurve(Ns,conf::repeat);

    lcurve.Generate(trainset, testset);

    arma::mat test_errors_ = lcurve.test_errors_;

    arma::mat Nss = arma::conv_to<arma::mat>::from(Ns);
    arma::mat results = arma::join_vert(Nss, test_errors_);

    utils::Save(conf::dir_lci/"nnc_wine.csv", results);
}

void real_class ( )
{
    utils::data::classification::Dataset dataset;
    dataset.Load("datasets/winequality-white.csv");

    utils::data::classification::Dataset trainset, testset;
    utils::data::StratifiedSplit(dataset, trainset, testset, 0.2);

    arma::irowvec Ns = arma::regspace<arma::irowvec>(2,10,3900);

    src::classification::LCurve<algo::classification::QDC<>,
                                  mlpack::Accuracy> lcurve(Ns,conf::repeat);

    lcurve.Generate(trainset, testset,1.);

    arma::mat test_errors_ = lcurve.test_errors_;

    arma::mat Nss = arma::conv_to<arma::mat>::from(Ns);
    arma::mat results = arma::join_vert(Nss, test_errors_);

    utils::Save(conf::dir_lci/"ldc_wine.csv", results);
}

void real_reg ( )
{
  utils::data::regression::Dataset dataset;
  arma::uvec ins = {0,1,2,3,4,5,6,7,9};
  arma::uvec outs =  {8};
  dataset.Load("datasets/housing.csv",ins,outs,true,true);

  utils::data::regression::Dataset trainset, testset;
  utils::data::Split(dataset, trainset, testset, 0.2);

  utils::data::regression::Transformer scaler(trainset);

  trainset = scaler.Trans(trainset);
  testset  = scaler.Trans(testset);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(20,10,1000);

  src::regression::LCurve<mlpack::LinearRegression,
                                mlpack::Accuracy> lcurve(Ns,conf::repeat);

  lcurve.Generate(trainset, testset);

  arma::mat test_errors_ = lcurve.test_errors_;

  arma::mat Nss = arma::conv_to<arma::mat>::from(Ns);
  arma::mat results = arma::join_vert(Nss, test_errors_);

  utils::Save(conf::dir_lci/"housing.csv", results);
}

};
};
