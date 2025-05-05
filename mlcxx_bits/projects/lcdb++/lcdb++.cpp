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

void run_jobs ( CLIStore& conf ) 
{
  std::filesystem::create_directories(lcdb::path);

  lcdb::DATASET data(conf.Get<size_t>("id")); 

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
          auto curve = lcurve::LCurve<T,DATASET,O,J,float>::Load(
              lcdb::path/(conf.GenName({"id","hpt","split","algo",
                                        "loss","samp","reps"})+".bin")) ;
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
          {
            curve->Generate();
          }

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
    Ns = arma::regspace<arma::Row<size_t>>(10,1,
          size_t((1.-lcdb::splitsize)*data.size_));
  else
    Ns = arma::regspace<arma::Row<size_t>>(1,1,
        size_t((1.-lcdb::splitsize)*data.size_));
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

            if ( conf.Get<bool>("split") )
            {
              lcdb::DATASET trainset,testset;
              data::Split(data,trainset,testset,lcdb::splitsize);
              lcurve::LCurve<T,DATASET,O,J,FIDEL> 
                curve(trainset,testset,Ns,conf.Get<size_t>("reps"),
                      true,true,lcdb::path,
                                        conf.GenName({"id","hpt","split","algo",
                                                      "loss","samp","reps"}));

              curve.Save(conf.GenName({"id","hpt","split","algo",
                                     "loss","samp","reps"})+".bin");
            }
            else
            {
              lcurve::LCurve<T,DATASET,O,J,FIDEL> 
                curve(data,Ns,conf.Get<size_t>("reps"), true,true,lcdb::path,
                                        conf.GenName({"id","hpt","split","algo",
                                                      "loss","samp","reps"}));

              curve.Save(conf.GenName({"id","hpt","split","algo",
                                     "loss","samp","reps"})+".bin");
          }
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

  conf.Register<size_t>("id",11,{11,37});
  conf.Register<bool>("hpt",false,{true,false});
  conf.Register<bool>("split",false,{true,false});
  conf.Register<std::string>("state","init");
  conf.Register<std::string>("algo","nmc",
      {"nmc","nnc","ldc","qdc","lreg","lsvc","gsvc","adab","rfor","dt"});
      /* {"nmc","nnc"}); */
  conf.Register<std::string>("loss","acc",{"acc","auc","bri","crs"});
  conf.Register<std::string>("samp","rands",{"rands","add","boot"});
  conf.Register<size_t>("reps",1);
  conf.Register<size_t>("seed",24);
  conf.Parse(argc,argv);
  /* conf.Set("split",false); */

  if ( conf.Get<std::string>("state") == "init" )
  {

    auto id_vals = conf.GetOptions<size_t>("id");
    auto algo_vals = conf.GetOptions<std::string>("algo");
    auto samp_vals = conf.GetOptions<std::string>("samp");
    auto loss_vals = conf.GetOptions<std::string>("loss");
    auto split_vals = conf.GetOptions<bool>("split");
    auto hpt_vals = conf.GetOptions<bool>("hpt");

    for (size_t id_val : id_vals)
      for (std::string algo_val : algo_vals)
        for (std::string samp_val : samp_vals)
          for (std::string loss_val : loss_vals)
            for (bool hpt_val : hpt_vals)
              for (bool split_val : split_vals)
              {
                conf.Set("id",id_val);
                conf.Set("algo",algo_val);
                conf.Set("samp",samp_val);
                conf.Set("loss",loss_val);
                conf.Set("hpt",hpt_val);
                conf.Set("split",split_val);
                conf.Print();
                init_jobs ( CLIStore::getInstance() );
              }
  }
  else if ( conf.Get<std::string>("state") == "run" )
  {
    conf.Print();
    run_jobs ( CLIStore::getInstance() );
  }

  arma::wall_clock timer;
  timer.tic();

  PRINT_TIME(timer.toc());
  return 0;

}
