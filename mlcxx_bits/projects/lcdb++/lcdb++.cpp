/**
 * @file lcdb++.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for lcdb++ generation
 *
 */

#define DTYPE double

#include <headers.h>
#include "config.h"

using namespace lcdb;

// Define aliases



void run_jobs ( CLIStore& conf ) 
{
  std::filesystem::create_directories(lcdb::path);

  lcdb::DATASET data(conf.Get<size_t>("id")); 
  arma::Row<size_t> Ns;
  if ( conf.Get<bool>("hpt") )
    Ns = arma::regspace<arma::Row<size_t>>(10,1,100);
  else
    Ns = arma::regspace<arma::Row<size_t>>(1,10,size_t(0.9*data.size_));

  try
  {
    lcdb::MODS _model = lcdb::models.at(conf.Get<std::string>("algo"));
    lcdb::LOSS _loss = lcdb::losses.at(conf.Get<std::string>("loss"));
    lcdb::SMPS _samp = lcdb::samples.at(conf.Get<std::string>("samp"));

    std::visit([&](auto t) {
      using T = typename decltype(t)::type;

      std::visit([&](auto o) {
        using O = typename decltype(o)::type;

        std::visit([&](auto j) {
          using J = typename decltype(j)::type;
          auto curve = lcurve::LCurve<MODEL,DATASET,SAMPLE,LOSS>::Load
                          (conf.GenName()+".bin");
          if (conf.Get<bool>("hpt"))
          {
            lcdb::HptSet hptset = lcdb::get_hptset(
                                                conf.Get<std::string>("algo"));

            std::visit([&](auto&& vec) 
            {
              curve->Generate(lcdb::vsize,vec);
            }, hptset);
          }
          else 
            curve.Generate();
        }, _loss);
      }, _samp);
    }, _model);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << "\n";
  }
  
}

void init_jobs ( CLIStore& conf ) 
{
  std::filesystem::create_directories(lcdb::path);

  lcdb::DATASET data(conf.Get<size_t>("id")); 
  arma::Row<size_t> Ns;
  if ( conf.Get<bool>("hpt") )
    Ns = arma::regspace<arma::Row<size_t>>(10,1,100);
  else
    Ns = arma::regspace<arma::Row<size_t>>(1,10,size_t(0.9*data.size_));

  try
  {
    lcdb::MODS _model = lcdb::models.at(conf.Get<std::string>("algo"));
    lcdb::LOSS _loss = lcdb::losses.at(conf.Get<std::string>("loss"));
    lcdb::SMPS _samp = lcdb::samples.at(conf.Get<std::string>("samp"));

    std::visit([&](auto t) {
      using T = typename decltype(t)::type;

      std::visit([&](auto o) {
        using O = typename decltype(o)::type;

        std::visit([&](auto j) {
          using J = typename decltype(j)::type;
          lcurve::LCurve<T,
                         DATASET,
                         O,
                         J,float> curve(data,Ns,conf.Get<size_t>("reps"),
                                        true,true,lcdb::path);
          curve.Save(conf.GenName());
        }, _loss);
      }, _samp);
    }, _model);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << "\n";
  }
}

int main(int argc, char* argv[]) 
{

  auto& conf = CLIStore::getInstance();

  conf.Register<size_t>("id",11);
  conf.Register<bool>("hpt",false);
  conf.Register<bool>("split",false);
  conf.Register<std::string>("state","init");
  conf.Register<std::string>("algo","nmc");
  conf.Register<std::string>("loss","acc");
  conf.Register<std::string>("samp","rands");
  conf.Register<size_t>("reps",1);
  conf.Register<size_t>("seed",24);

  conf.Parse(argc,argv);
  conf.Print();

  if ( conf.Get<std::string>("state") == "init" )
    init_jobs ( CLIStore::getInstance() );
  else if ( conf.Get<std::string>("state") == "run" )
    run_jobs ( CLIStore::getInstance() );
  else if ( conf.Get<std::string>("state") == "save" )
    save_res ( CLIStore::getInstance() );

  arma::wall_clock timer;
  timer.tic();

  PRINT_TIME(timer.toc());
  return 0;

}
