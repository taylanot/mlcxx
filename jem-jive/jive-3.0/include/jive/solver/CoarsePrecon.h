
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

#ifndef JIVE_SOLVER_COARSEPRECON_H
#define JIVE_SOLVER_COARSEPRECON_H

#include <jive/solver/Preconditioner.h>


JIVE_BEGIN_PACKAGE( solver )


class CoarseMatrix;


//-----------------------------------------------------------------------
//   class CoarsePrecon
//-----------------------------------------------------------------------


class CoarsePrecon : public Preconditioner
{
 public:

  JEM_DECLARE_CLASS       ( CoarsePrecon, Preconditioner );

  static const char*        TYPE_NAME;


                            CoarsePrecon

    ( const String&           name,
      Ref<CoarseMatrix>       cmat );

  virtual void              resetEvents     ()       override;

  virtual Shape             shape           () const override;

  virtual void              start           ()       override;
  virtual void              finish          ()       override;
  virtual void              update          ()       override;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs )            const override;

  virtual void              configure

    ( const Properties&       props )                override;

  virtual void              getConfig

    ( const Properties&       props )          const override;

  virtual bool              hasTrait

    ( const String&           trait )          const override;

  virtual Constraints*      getConstraints  () const override;

  static Ref<Precon>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       params,
      const Properties&       globdat );

  static void               declare         ();


 protected:

  virtual                  ~CoarsePrecon    ();


 private:

  void                      connect_        ();
  void                      update_         ();

  void                      valuesChanged_  ();
  void                      structChanged_  ();

  void                      syncEvents_     ();

  void                      setEvents_

    ( int                     events );


 private:

  static const int          NEW_VALUES_;
  static const int          NEW_STRUCT_;

  Ref<GramSchmidt>          grams_;
  Ref<CoarseMatrix>         coarseMat_;

  Matrix                    qFactor_;
  Matrix                    rFactor_;

  int                       events_;
  idx_t                     started_;

};


JIVE_END_PACKAGE( solver )

#endif
