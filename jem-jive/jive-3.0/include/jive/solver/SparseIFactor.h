
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

#ifndef JIVE_SOLVER_SPARSEIFACTOR_H
#define JIVE_SOLVER_SPARSEIFACTOR_H

#include <jem/base/Flags.h>
#include <jive/SparseMatrix.h>
#include <jive/solver/import.h>
#include <jive/solver/Preconditioner.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class SparseIFactor
//-----------------------------------------------------------------------


class SparseIFactor : public Preconditioner
{
 public:

  JEM_DECLARE_CLASS       ( SparseIFactor, Preconditioner );

  typedef VectorExchanger   Exchanger;

  static const double       ZERO_THRESHOLD;

  enum                      Option
  {
                              REORDER = 1 << 0
  };

  typedef
    jem::Flags<Option>      Options;


                            SparseIFactor

    ( const String&           name,
      Ref<AbstractMatrix>     matrix,
      Ref<Constraints>        cons = nullptr );

  virtual void              resetEvents       ()       override;
  virtual Shape             shape             () const override;

  virtual void              start             ()       override;
  virtual void              finish            ()       override;
  virtual void              update            ()       override;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs )              const override;

  virtual void              configure

    ( const Properties&       props )                  override;

  virtual void              getConfig

    ( const Properties&       props )            const override;

  virtual bool              hasTrait

    ( const String&           trait )            const override;

  virtual Constraints*      getConstraints    () const override;

  void                      setOption

    ( Option                  option,
      bool                    yesno = true );

  void                      setOptions

    ( Options                 options );

  inline Options            getOptions        () const noexcept;

  void                      setDropTol

    ( double                  tol );

  inline double             getDropTol        () const;

  void                      setDiagShift

    ( double                  dshift );

  inline double             getDiagShift      () const;

  void                      setZeroThreshold

    ( double                  eps );

  inline double             getZeroThreshold  () const;

  void                      setExchangeMode

    ( int                     xmode );

  inline int                getExchangeMode   () const;

  static bool               decodeParams

    ( Ref<AbstractMatrix>&    matrx,
      Ref<Constraints>&       cons,
      const Properties&       params );


 protected:

  virtual                  ~SparseIFactor     ();

  virtual void              factor_

    ( int                     events,
      const SparseMatrix&     matrix,
      const BoolVector&       mask,
      const IdxVector&        iperm )                  = 0;

  virtual void              solve_

    ( const Vector&           lhs,
      const Vector&           rhs  )             const = 0;

  void                      setParam_

    ( int&                    param,
      int                     value );

  void                      setParam_

    ( idx_t&                  param,
      idx_t                   value );

  void                      setParam_

    ( double&                 param,
      double                  value );

  void                      setEvents_

    ( int                     events );


 protected:

  static const int          NEW_VALUES_;
  static const int          NEW_STRUCT_;
  static const int          NEW_CONFIG_;

  Options                   options_;
  idx_t                     maxFill_;
  double                    droptol_;
  double                    dshift_;
  double                    zeroThreshold_;
  bool                      symmetric_;


 private:

  void                      connect_          ();
  void                      update_           ();
  void                      valuesChanged_    ();
  void                      structChanged_    ();
  void                      syncEvents_       ();


 private:

  Ref<AbstractMatrix>       matrix_;
  Ref<Constraints>          cons_;
  Ref<Exchanger>            exchanger_;

  IdxVector                 dofPerm_;
  IdxVector                 borderDofs_;
  Vector                    borderDiag_;

  int                       xmode_;
  int                       events_;
  idx_t                     started_;

};


JEM_DEFINE_FLAG_OPS( SparseIFactor::Options )


//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


inline SparseIFactor::Options
  SparseIFactor::getOptions () const noexcept
{
  return options_;
}


//-----------------------------------------------------------------------
//   getDropTol
//-----------------------------------------------------------------------


inline double SparseIFactor::getDropTol () const
{
  return droptol_;
}


//-----------------------------------------------------------------------
//   getDiagShift
//-----------------------------------------------------------------------


inline double SparseIFactor::getDiagShift () const
{
  return dshift_;
}


//-----------------------------------------------------------------------
//   getExchangeMode
//-----------------------------------------------------------------------


inline int SparseIFactor::getExchangeMode () const
{
  return xmode_;
}


//-----------------------------------------------------------------------
//   getZeroThreshold
//-----------------------------------------------------------------------


inline double SparseIFactor::getZeroThreshold () const
{
  return zeroThreshold_;
}


JIVE_END_PACKAGE( solver )

#endif
