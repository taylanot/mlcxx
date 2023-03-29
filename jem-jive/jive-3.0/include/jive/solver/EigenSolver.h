
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

#ifndef JIVE_SOLVER_EIGENSOLVER_H
#define JIVE_SOLVER_EIGENSOLVER_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class EigenSolver
//-----------------------------------------------------------------------


class EigenSolver : public NamedObject
{
 public:

  JEM_DECLARE_CLASS       ( EigenSolver, NamedObject );

  static const double       PRECISION;


  explicit                  EigenSolver

    ( const String&           name = "" );

  virtual void              getInfo

    ( const Properties&       info )         const;

  virtual void              configure

    ( const Properties&       props );

  virtual void              getConfig

    ( const Properties&       props )        const;

  virtual void              getEigenValues

    ( const Matrix&           evals,
      idx_t                   smallCount )         = 0;

  virtual void              getEigenVectors

    ( const Matrix&           evals,
      const Cubix&            evecs,
      idx_t                   smallCount )         = 0;

  virtual void              getRealEigenValues

    ( const Vector&           evals,
      idx_t                   smallCount );

  virtual void              getRealEigenVectors

    ( const Vector&           evals,
      const Matrix&           evecs,
      idx_t                   smallCount );

  virtual void              setPrecision

    ( double                  eps )                = 0;

  virtual double            getPrecision  () const = 0;
  virtual String            getContext    () const override;


 protected:

  virtual                  ~EigenSolver   ();


 protected:

  String                    myName_;

};


JIVE_END_PACKAGE( solver )

#endif
