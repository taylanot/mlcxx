
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_SOLVER_VERBOSESOLVER_H
#define JIVE_SOLVER_VERBOSESOLVER_H

#include <jem/util/Timer.h>
#include <jive/solver/import.h>
#include <jive/solver/Solver.h>


JIVE_BEGIN_PACKAGE( solver )


class DirectSolver;
class IterativeSolver;


//-----------------------------------------------------------------------
//   class VerboseSolver
//-----------------------------------------------------------------------


class VerboseSolver : public Solver
{
 public:

  JEM_DECLARE_CLASS       ( VerboseSolver, Solver );

  typedef PrintWriter       Printer;

  static const char*        TYPE_NAME;


                            VerboseSolver

    ( const String&           name,
      const Ref<Solver>&      solver,
      const Ref<Printer>&     out = nullptr );

  virtual void              start             ()       override;
  virtual void              finish            ()       override;

  virtual void              improve

    ( const Vector&           lhs,
      const Vector&           rhs )                    override;

  virtual void              getInfo

    ( const Properties&       info )             const override;

  virtual void              configure

    ( const Properties&       props )                  override;

  virtual void              getConfig

    ( const Properties&       props )            const override;

  virtual void              setMode

    ( int                     mode )                   override;

  virtual int               getMode           () const override;

  virtual void              setPrecision

    ( double                  eps )                    override;

  virtual double            getPrecision      () const override;
  virtual AbstractMatrix*   getMatrix         () const override;
  virtual Constraints*      getConstraints    () const override;

  void                      setPrintInterval

    ( double                  dt );

  double                    getPrintInterval  () const;

  void                      setNoiseLevel

    ( int                     nlevel );

  int                       getNoiseLevel     () const;

  static Ref<Solver>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       params,
      const Properties&       globdat );

  static void               declare           ();


 protected:

  virtual                  ~VerboseSolver     ();

  virtual void              begin_            ();
  virtual void              end_              ();


 protected:

  Ref<Solver>               solver_;
  String                    solverName_;

  Ref<Printer>              output_;

  double                    printInterval_;
  int                       noiseLevel_;

  jem::util::Timer          timer_;
  double                    lastTime_;


 private:

  void                      start_            ();
  void                      stop_             ();

};


//-----------------------------------------------------------------------
//   class DirectVerboseSolver
//-----------------------------------------------------------------------


class DirectVerboseSolver : public VerboseSolver
{
 public:

  JEM_DECLARE_CLASS       ( DirectVerboseSolver, VerboseSolver );


                            DirectVerboseSolver

    ( const String&           name,
      Ref<DirectSolver>       solver,
      const Ref<Printer>&     out = nullptr );


 protected:

  virtual                  ~DirectVerboseSolver ();

  virtual void              begin_              () override;
  virtual void              end_                () override;


 protected:

  enum                      Phase_
  {
                              INACTIVE_,
                              FACTORING_,
                              SOLVING_
  };


  Ref<DirectSolver>         solver_;
  Phase_                    phase_;
  idx_t                     zeroPivotCount_;


 private:

  void                      factorHandler_

    ( idx_t                   done  );

  void                      zeroPivotHandler_

    ( double                  pivot );

  void                      solveHandler_

    ( double                  err   );

};


//-----------------------------------------------------------------------
//   class IterativeVerboseSolver
//-----------------------------------------------------------------------


class IterativeVerboseSolver : public VerboseSolver
{
 public:

  JEM_DECLARE_CLASS       ( IterativeVerboseSolver, VerboseSolver );


                            IterativeVerboseSolver

    ( const String&           name,
      Ref<IterativeSolver>    solver,
      const Ref<Printer>&     out = nullptr );


 protected:

  virtual                  ~IterativeVerboseSolver  ();

  virtual void              begin_                  () override;
  virtual void              end_                    () override;


 protected:

  Ref<IterativeSolver>      solver_;
  idx_t                     lastIter_;


 private:

  void                      nextIterHandler_

    ( idx_t                   iiter,
      double                  err );

  void                      restartHandler_

    ( idx_t                   iiter,
      double                  err );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<VerboseSolver>          newVerboseSolver

  ( const Ref<Solver>&         solver,
    const Ref<PrintWriter>&    out = nullptr );

Ref<VerboseSolver>          newVerboseSolver

  ( const String&              name,
    const Ref<Solver>&         solver,
    const Ref<PrintWriter>&    out = nullptr );


JIVE_END_PACKAGE( solver )

#endif
