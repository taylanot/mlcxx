
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverInfo.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/Preconditioner.h>
#include <jive/solver/IterativeSolverException.h>
#include <jive/solver/CG.h>


JEM_DEFINE_CLASS( jive::solver::CG );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class CG
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  CG::TYPE_NAME = "CG";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


CG::CG

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<VectorSpace>     vspace,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  precon ) :

    Super ( name, matrix, vspace, cons, precon )

{
  using jem::System;
  using jive::util::joinNames;


  if ( ! matrix->isSymmetric() )
  {
    print ( System::warn(), myName_, " : non-symmetric matrix\n" );
  }

  if ( ! precon_->isSymmetric() )
  {
    print ( System::warn(), myName_,
            " : non-symmetric preconditioner\n" );
  }
}


CG::~CG ()
{}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void CG::getInfo ( const Properties& info ) const
{
  Super::getInfo ( info );

  info.set ( SolverInfo::TYPE_NAME, TYPE_NAME );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> CG::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Ref<AbstractMatrix>  matrix;
  Ref<AbstractMatrix>  precon;
  Ref<Constraints>     cons;
  Ref<VectorSpace>     vspace;


  if ( ! Super::decodeParams( matrix, vspace, cons, params ) )
  {
    return nullptr;
  }

  if ( ! params.find( precon, SolverParams::PRECONDITIONER ) )
  {
    precon = newPrecon (
      joinNames ( name, PropNames::PRECONDITIONER ),
      conf, props, params, globdat
    );
  }

  return jem::newInstance<Self> ( name,   matrix,
                                  vspace, cons, precon );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void CG::declare ()
{
  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------


void CG::solve_

  ( idx_t&         iiter,
    double&        error,
    double         rscale,
    const Vector&  lhs,
    const Vector&  rhs )

{
  using jem::equal;
  using jem::numeric::axpy;

  const VectorSpace&     vspace  = * vspace_;
  const AbstractMatrix&  matrix  = * matrix_;
  AbstractMatrix&        precon  = * precon_;

  const double  DELTA    = 1.0e-3;

  const idx_t   dofCount = lhs.size ();

  Matrix        rvt      ( dofCount, 3 );
  Vector        products ( 3 );

  Vector        r ( rvt[0] );
  Vector        v ( rvt[1] );
  Vector        t ( rvt[2] );

  Vector        p ( dofCount );
  Vector        z ( dofCount );

  double        alpha, beta, gamma, delta, ksi;
  double        rho,   rhoPrev;
  double        znorm, zscale, zscale0;


  lhs = 0.0;
  r   = rhs;

  precon.matmul ( z, r );

  p = z;

  matrix.matmul ( v, z );

  t = v;

  vspace.project ( products, z, rvt );

  rho = rhoPrev = products[0];
  ksi = gamma   = products[1];

  if ( rho < 0.0 )
  {
    throw IterativeSolverException (
      getContext (),
      "non-positive definite matrix or preconditioner",
      iiter,
      error
    );
  }

  znorm  = std::sqrt( rho );
  zscale = zscale0 = error / znorm;

  while ( iiter < maxIter_ )
  {
    alpha   = rho / ksi;
    rhoPrev = rho;

    axpy ( lhs,  alpha, p );
    axpy ( r,   -alpha, t );

    iiter++;

    precon.matmul  ( z, r );
    matrix.matmul  ( v, z );
    vspace.project ( products, z, rvt );

    rho   = products[0];
    gamma = products[1];
    delta = products[2];
    beta  = rho / rhoPrev;
    ksi   = gamma + 2.0 * beta * delta + beta * beta * ksi;

    axpy  ( p, z, beta, p );
    axpy  ( t, v, beta, t );

    znorm = std::sqrt ( std::fabs( rho ) );

    nextIterEvent.emit ( iiter, zscale * znorm, *this );

    if ( (zscale * znorm) <= (0.95 * precision_) )
    {
      getResidual ( r, lhs, rhs );

      error = rscale * vspace.norm2( r );

      nextIterEvent.emit ( iiter, error, *this );

      if ( error <= precision_ || error > MAX_RESIDUAL )
      {
        return;
      }

      zscale = error / znorm;

      if ( zscale0 < DELTA * zscale )
      {
        return;
      }
    }

    if ( rho < 0.0 )
    {
      throw IterativeSolverException (
        getContext (),
        "non-positive definite matrix or preconditioner",
        iiter,
        error
      );
    }

    testCancelled_ ();
  }
}


JIVE_END_PACKAGE( solver )
