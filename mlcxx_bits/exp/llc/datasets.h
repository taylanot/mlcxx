/**
 * @file datasets.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef LLC_DATASETS_H 
#define LLC_DATASETS_H

namespace experiments {
namespace llc {

//-----------------------------------------------------------------------------
// CLASSIFICATION
//-----------------------------------------------------------------------------
void classification( const size_t& N )
{
  
  std::filesystem::path dir, filename; 

  // 2D-Gaussian Blobs that are chaning distance between them
  {
    PRINT("2D-Gaussian Blobs that are chaning distance between them...")
    dir =  conf::data_class_dir / conf::class_set[0];

    std::filesystem::create_directories(dir);

    utils::data::classification::Dataset dataset(conf::Dc, N, conf::Nc);

    for ( size_t i=0; i<conf::Ndelta; i++ )
    {
      dataset._2classgauss (conf::mean, conf::mean, 1., conf::delta(i));
      filename =  dir / (std::to_string(i)+".bin");
      dataset.Save(filename);
    }

    PRINT("2D-Gaussian Blobs that are chaning distance between them...[DONE]")
  }
 // Banana dataset with changing distance
  {
    PRINT("Banana Changing distance...")
    dir =  conf::data_class_dir / conf::class_set[1];

    std::filesystem::create_directories(dir);

    utils::data::classification::Dataset dataset(conf::Dc, N, conf::Nc);

    for ( size_t i=0; i<conf::Ndelta; i++ )
    {
      dataset._banana(conf::delta(i));
      filename =  dir / (std::to_string(i)+".bin");
      dataset.Save(filename);
    }
    PRINT("Banana Changing distance...[DONE]")
  }

  //// 1D Dipping data this is just one dataset 
  //{
  //  PRINT("1D Dipping...")
  //  dir =  conf::data_class_dir / conf::class_set[2];

  //  std::filesystem::create_directories(dir);

  //  utils::data::classification::Dataset dataset(conf::Ddip, N, conf::Nc);
  //  std::string type = "Dipping";
  //  dataset.Generate(type);
  //  filename =  dir/ (std::to_string(0)+".bin");
  //  dataset.Save(filename);
  //  PRINT("1D Dipping...[DONE]")
  //}

  //// Delayed Dipping for increasing proble dimensionality
  //{
  //  PRINT("Delayed Dipping changing dimension...")
  //  dir =  conf::data_class_dir / conf::class_set[2];

  //  std::filesystem::create_directories(dir);
  //  for ( size_t d=2; d<=conf::Nhyper; d++ )
  //  {
  //    utils::data::classification::Dataset dataset(d, N, conf::Nc);
  //    dataset._dipping(conf::r(0),1.);
  //    filename =  dir / (std::to_string(d-1)+".bin");
  //    dataset.Save(filename);
  //  }
  //  PRINT("Delayed Dipping changing dimension...[DONE]")
  //}

  // Delayed Dipping for increasing radius of the ring
  {
    PRINT("Delayed Dipping changing radius...")
    dir =  conf::data_class_dir / conf::class_set[2];

    std::filesystem::create_directories(dir);
    
    for ( size_t i=0; i<conf::Nhyper; i++ )
    {
      utils::data::classification::Dataset dataset(conf::Dc, N, conf::Nc);
      dataset._dipping(conf::r(i),1.);
      filename =  dir / (std::to_string(i)+".bin");
      dataset.Save(filename);
    }
    PRINT("Delayed Dipping changing radius...[DONE]")
  }

  
}

//-----------------------------------------------------------------------------
// REGRESSION
//-----------------------------------------------------------------------------

void regression( const size_t& N )
{
  std::filesystem::path dir, filename, addition; 

    // Linear dataset for changing noise levels
  {
    PRINT("Linear dataset changing noise levels...")
    dir =  conf::data_reg_dir / conf::reg_set[1] ;

    std::filesystem::create_directories(dir);


    utils::data::regression::Dataset dataset(conf::D, conf::N);

    for ( size_t i=0; i<conf::Neps; i++ )
    {
      dataset.Generate(1.,0.,"Linear",conf::eps(i));
      filename =  dir / (std::to_string(i)+".bin");
      dataset.Save(filename);
    }
    PRINT("Linear dataset changing noise levels...[DONE]")
  }

  // Linear dataset for changing dimensionality
  {
    PRINT("Linear dataset changing dimensionality...")
    dir =  conf::data_reg_dir / conf::reg_set[0];

    std::filesystem::create_directories(dir);

    for ( size_t d=2; d<=conf::Nd; d++ )
    {
      utils::data::regression::Dataset dataset(d, N);
      dataset.Generate(1.,0.,"Linear",1.);
      filename =  dir / (std::to_string(d-2)+".bin");
      dataset.Save(filename);
    }
    PRINT("Linear dataset changing dimensionality...[DONE]")
  }

  // Sine dataset for changing noise levels
  {
    PRINT("Sine dataset changing noise...")
    dir =  conf::data_reg_dir / conf::reg_set[3];

    std::filesystem::create_directories(dir);

    utils::data::regression::Dataset dataset(conf::D, N);

    for ( size_t i=0; i<conf::Neps; i++ )
    {
      dataset.Generate(1.,0.,"Sine",conf::eps(i));
      filename =  dir / (std::to_string(i)+".bin");
      dataset.Save(filename);
    }
    PRINT("Sine dataset changing noise...[DONE]")
  }

  // Sinc dataset for changing noise levels
  {
    PRINT("Sinc dataset changing noise...")
    dir =  conf::data_reg_dir/ conf::reg_set[2];

    std::filesystem::create_directories(dir);

    utils::data::regression::Dataset dataset(conf::D, N);


    for ( size_t i=0; i<conf::Neps; i++ )
    {
      dataset.Generate(1.,0.,"Sinc",conf::eps(i));
      filename =  dir / (std::to_string(i)+".bin");
      dataset.Save(filename);
    }
    PRINT("Sinc dataset changing noise...[DONE]")
  }


  // Gaussian Process datasets
  {
    PRINT("GP dataset changing noise variance...")

    dir =  conf::data_reg_dir / conf::reg_set[4];

    for ( size_t c=0; c<conf::repeat_gp; c++ )
    {
      addition = dir / std::to_string(c);
      std::filesystem::create_directories(addition);
      
      for ( size_t i=0; i<conf::Nlambda_gp; i++ )
      {
        arma::mat inputs = arma::sign(arma::randn(conf::D_gp,N));
        arma::mat labels;
        algo::regression::GaussianProcess<mlpack::GaussianKernel> 
                                      GPteach(double(std::sqrt(conf::D_gp)));
        GPteach.Lambda(conf::lambda_gp_teach(i));
        GPteach.SamplePrior(1, inputs, labels);
        filename = addition / (std::to_string(i)+".bin");
        utils::Save(filename, inputs, labels);
      }
    }
    PRINT("GP dataset changing noise variance...[DONE]")
  }


  
}

void datasets()
{
  PRINT("Creating llc datasets...")
  classification(conf::N); 
  PRINT("Classification datasets are done!")
  //regression(conf::N); 
  //PRINT("Regression datasets are done!")
  }

} // namespace llc
} // namespace exp
#endif 

