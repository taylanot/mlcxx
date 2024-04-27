/**
 * @file main_func.h
 * @author Ozgur Taylan Turan
 *
 *
 */

#ifndef MAIN_FUNC_H
#define MAIN_FUNC_H

//-----------------------------------------------------------------------------
//   func_run
//-----------------------------------------------------------------------------

void func_run( char* argv )
{
  //// REGISTER YOUR PREDEFINED FUNCTIONS HERE
  typedef void (*FuncType)( );
  std::map<std::string, FuncType> m;

  // LLC-PAPER 
  m["llc-datasets"] = experiments::llc::datasets;
  m["llc-lcgen"]  = experiments::llc::lcgen;

  m["llc-ldc"]   = experiments::llc::gen_ldc;
  m["llc-qdc"]   = experiments::llc::gen_qdc;
  m["llc-nnc"]   = experiments::llc::gen_nnc;
  m["llc-nmc"]   = experiments::llc::gen_nmc;

  m["llc-saw"]   = experiments::llc::gen_saw;
  m["llc-lin"]   = experiments::llc::gen_lin;
  m["llc-dgp"]   = experiments::llc::gen_dgp;
  m["llc-nn-adam"]   = experiments::llc::gen_nn_adam;
  m["llc-sgd"]   = experiments::llc::gen_nn_sgd;
  m["llc-lkr"]   = experiments::llc::gen_lkr;
  m["llc-gkr"]   = experiments::llc::gen_gkr;


  m["llc-curvefit"] = experiments::llc::run_curve_fit;
  m["llc-curvefitat"] = experiments::llc::run_curve_fit_at;
  m["llc-curvefit-pred"] = experiments::llc::run_curve_fit_pred;
  m["llc-spkrfit2-before"] = experiments::llc::run_spkr2_fit_before;
  m["llc-spkrfit2-after"] = experiments::llc::run_spkr2_fit_after;

  
  if (experiments::llc::conf::subset)
    m["llc-spkrfit2"] = experiments::llc::subset_run_spkr2_fit;
  else
    m["llc-spkrfit2"] = experiments::llc::run_spkr2_fit;

  // YAWLC 
  m["yawlc-linhes"] = experiments::yawlc::linhes;
  m["yawlc-linheslc"] = experiments::yawlc::linheslc;
  m["yawlc-stats"] = experiments::yawlc::stats;
  m["yawlc-linear"] = experiments::yawlc::linearmodel;

  m["yawlc-gd"] = experiments::yawlc::gd;
  m["yawlc-sgd"] = experiments::yawlc::sgd;

  // LCI
  m["lci-lin"] = experiments::lci::lin;
  m["lci-lin2"] = experiments::lci::lin2;
  m["lci-lin3"] = experiments::lci::lin3;
  m["lci-class"] = experiments::lci::real_class;
  m["lci-reg"] = experiments::lci::real_reg;

  BOOST_ASSERT_MSG((m.find(argv) != m.end()),
   "You should register your function to 'main_func.h'!!");
  m[argv]();

}

#endif

