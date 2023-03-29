
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

#ifndef JIVE_SOLVER_POINTCONSTRAINER_H
#define JIVE_SOLVER_POINTCONSTRAINER_H

#include <jive/solver/import.h>
#include <jive/solver/Constrainer.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class PointConstrainer
//-----------------------------------------------------------------------


class PointConstrainer : public Constrainer
{
 public:

  JEM_DECLARE_CLASS       ( PointConstrainer, Constrainer );

  static const char*        TYPE_NAME;
  static const char*        MODE_PROP;


                            PointConstrainer

    ( const String&           name,
      Ref<Constraints>        cons,
      Ref<AbstractMatrix>     input,
      Ref<SparseMatrixObj>    output = nullptr );

  virtual void              update                ()       override;
  virtual bool              updated               () const override;

  virtual void              getRhs

    ( const Vector&           g,
      const Vector&           f )                    const override;

  virtual void              getLhs

    ( const Vector&           x,
      const Vector&           y )                    const override;

  virtual void              initLhs

    ( const Vector&           x,
      const Vector&           y )                    const override;

  virtual Constraints*      getConstraints        () const override;
  virtual AbstractMatrix*   getInputMatrix        () const override;
  virtual AbstractMatrix*   getOutputMatrix       () const override;

  static Ref<Constrainer>   makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      Ref<Constraints>        cons,
      Ref<AbstractMatrix>     matrix );

  static void               declare               ();


 protected:

  virtual                  ~PointConstrainer      ();

  virtual void              doUpdate_             ()       = 0;


 protected:

  static const int          NEW_RVALUES_;
  static const int          NEW_SLAVE_DOFS_;
  static const int          NEW_MAT_VALUES_;
  static const int          NEW_MAT_STRUCT_;

  Ref<Constraints>          cons_;
  Ref<AbstractMatrix>       input_;
  Ref<SparseMatrixObj>      output_;

  BoolVector                dofMask_;
  IdxVector                 slaveDofs_;
  Vector                    rvalues_;
  Vector                    rhs_;

  int                       events_;


 private:

  void                      rvaluesChanged_       ();
  void                      slaveDofsChanged_     ();
  void                      matValuesChanged_     ();
  void                      matStructChanged_     ();


 private:

  bool                      warnFlag_;

};


//-----------------------------------------------------------------------
//   class DirectPointConstrainer
//-----------------------------------------------------------------------


class DirectPointConstrainer : public PointConstrainer
{
 public:

  JEM_DECLARE_CLASS       ( DirectPointConstrainer,
                            PointConstrainer );


                            DirectPointConstrainer

    ( const String&           name,
      Ref<Constraints>        cons,
      Ref<SparseMatrixObj>    matrix );


 protected:

  virtual                  ~DirectPointConstrainer ();

  virtual void              doUpdate_              () override;


 private:

  void                      updateRhs_             ();
  void                      restoreValues_         ();
  void                      updateValues_          ();
  void                      updateStructure_       ();

  void                      warnForSingularMatrix_ ();

 private:

  Ref<SparseMatrixObj>      zappedMatrix_;

  IdxVector                 zappedEntries_;
  IdxVector                 zappedDiagEntries_;

  bool                      warnFlag_;

};


//-----------------------------------------------------------------------
//   FilterPointConstrainer
//-----------------------------------------------------------------------


class FilterPointConstrainer : public PointConstrainer
{
 public:

  JEM_DECLARE_CLASS       ( FilterPointConstrainer,
                            PointConstrainer );


                            FilterPointConstrainer

    ( const String&           name,
      Ref<Constraints>        cons,
      Ref<AbstractMatrix>     input,
      Ref<SparseMatrixObj>    output = nullptr );


 protected:

  virtual                  ~FilterPointConstrainer ();

  virtual void              doUpdate_              () override;


 private:

  void                      updateRhs_             ();
  void                      updateValues_          ();
  void                      updateStruct_          ();

};


JIVE_END_PACKAGE( solver )

#endif
