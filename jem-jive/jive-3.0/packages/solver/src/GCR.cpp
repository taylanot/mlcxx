
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
#include <jem/base/limits.h>
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Event.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/declare.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/algebra/GramSchmidt.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverInfo.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/Preconditioner.h>
#include <jive/solver/IterativeSolverException.h>
#include <jive/solver/GMRESUtils.h>
#include <jive/solver/GCR.h>


JEM_DEFINE_CLASS( jive::solver::GCR );


JIVE_BEGIN_PACKAGE( solver )


using jem::System;


//=======================================================================
//   class GCR
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GCR::TYPE_NAME   = "GCR";
const idx_t  GCR::MAX_VECTORS = 100;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GCR::GCR

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<VectorSpace>     vspace,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  precon,
    Ref<GramSchmidt>     grams ) :

    Super  ( name, matrix, vspace, cons, precon ),
    grams_ ( grams )

{
  using jem::util::connect;
  using jive::util::joinNames;

  if ( ! grams_ )
  {
    grams_ = newGramSchmidt (
      joinNames  ( myName_, PropNames::GRAM_SCHMIDT ),
      Properties (),
      vspace
    );
  }

  vectCount_  = 0;
  maxVectors_ = MAX_VECTORS;

  connect ( precon_->newStructEvent, this, & Self::reset_ );
  connect ( precon_->newValuesEvent, this, & Self::reset_ );
  connect ( matrix ->newStructEvent, this, & Self::reset_ );
  connect ( matrix ->newValuesEvent, this, & Self::reset_ );
}


GCR::~GCR ()
{}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void GCR::getInfo ( const Properties& info ) const
{
  double  musage = 0.0;


  Super::getInfo ( info );

  info.find ( musage, SolverInfo::MEM_USAGE );

  musage += 2.0 * (double) sizeof(double)    *
                  (double) vVectors_.size(0) *
                  (double) vVectors_.size(1);

  info.set ( SolverInfo::TYPE_NAME, TYPE_NAME  );
  info.set ( SolverInfo::MEM_USAGE, musage     );
  info.set ( "vectorCount",         vectCount_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void GCR::configure ( const Properties& props )
{
  using jem::maxOf;

  Super ::configure ( props );
  grams_->configure ( props );

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    idx_t       count;
    bool        found;

    found = myProps.find ( count,
                           PropNames::MAX_VECTORS,
                           1, maxOf( count ) );

    if ( found )
    {
      setMaxVectors ( count );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void GCR::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super ::getConfig ( props );
  grams_->getConfig ( props );

  myProps.set ( PropNames::MAX_VECTORS, maxVectors_ );
}


//-----------------------------------------------------------------------
//   setMaxVectors
//-----------------------------------------------------------------------


void GCR::setMaxVectors ( idx_t count )
{
  JEM_PRECHECK ( count > 0 );

  if ( count < maxVectors_ )
  {
    reset_ ();
  }

  maxVectors_ = count;
}


//-----------------------------------------------------------------------
//   getMaxVectors
//-----------------------------------------------------------------------


idx_t GCR::getMaxVectors () const
{
  return maxVectors_;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> GCR::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Ref<AbstractMatrix>  matrix;
  Ref<AbstractMatrix>  precon;
  Ref<VectorSpace>     vspace;
  Ref<Constraints>     cons;
  Ref<GramSchmidt>     grams;


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

  grams = newGramSchmidt (
    joinNames ( name, PropNames::GRAM_SCHMIDT ),
    conf, props, vspace
  );

  return jem::newInstance<Self> ( name, matrix, vspace,
                                  cons, precon, grams );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void GCR::declare ()
{
  algebra::declareGSAlgorithms ();

  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}

//-----------------------------------------------------------------------
//   updatePrecon_
//-----------------------------------------------------------------------


void GCR::updatePrecon_ ()
{
  Super::updatePrecon_ ();

  vectCount_ = 0;
}


//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------


void GCR::solve_

  ( idx_t&         iiter,
    double&        error,
    double         rscale,
    const Vector&  lhs,
    const Vector&  rhs )

{
  using jem::equal;
  using jem::Float;
  using jem::numeric::axpy;
  using jem::numeric::matmul;

  const VectorSpace&     vspace = * vspace_;
  const AbstractMatrix&  matrix = * matrix_;
  AbstractMatrix&        precon = * precon_;

  const idx_t   dofCount = lhs.size ();

  Matrix&       vVecs    = vVectors_;
  Matrix&       qVecs    = qVectors_;

  Matrix        rv ( dofCount, 2 );
  Matrix        qz ( dofCount, 2 );

  Vector        r  ( rv[0] );
  Vector        v  ( rv[1] );
  Vector        q  ( qz[0] );
  Vector        z  ( qz[1] );

  Vector        a  ( maxVectors_ + 2 );

  double        x;

  idx_t         k;


  k = vectCount_;
  a = 0.0;
  r = rhs;

  if ( k == 0 )
  {
    lhs = 0.0;
  }
  else
  {
    grams_->orthogonalize ( a[slice(BEGIN,k)], r,
                            vVecs[slice(BEGIN,k)] );

    matmul ( z, qVecs[slice(BEGIN,k)], a[slice(BEGIN,k)] );

    precon.matmul  (    lhs, z   );
    getResidual    ( r, lhs, rhs );

    error = rscale * vspace.norm2( r );

    if ( error <= precision_ )
    {
      return;
    }

    lhs = z;
  }

  while ( k < maxVectors_ && iiter < maxIter_ )
  {
    q = r;

    precon.matmul ( z, q );
    matrix.matmul ( v, z );

    if ( k == 0 )
    {
      a[k] = a[k + 1] = vspace.norm2 ( v );
    }
    else
    {
      grams_->orthogonalize ( a[slice(BEGIN,k + 2)], v,
                              vVecs[slice(BEGIN,k)] );
    }

    // Check for a zero Krylov vector.

    if ( 0.1 * a[k + 1] <= Float::EPSILON * a[k] )
    {
      // Compute the best solution so far ...

      z = lhs;

      precon.matmul ( lhs, z );

      // ... and exit

      throw IterativeSolverException (
        getContext (),
        "zero krylov vector encountered "
        "(singular matrix or preconditioner?)",
        iiter,
        error
      );
    }

    if ( k > 0 )
    {
      matmul ( z, qVecs[slice(BEGIN,k)], a[slice(BEGIN,k)] );

      q -= z;
    }

    x  = 1.0 / a[k + 1];
    v *= x;
    q *= x;

    vspace.project ( a[slice(BEGIN,2)], r, rv );

    error = rscale * std::sqrt ( a[0] );
    x     = a[1];

    axpy ( lhs,  x, q );
    axpy ( r,   -x, v );

    // Compute the norm of the updated residual. If large roundoff
    // errors occur, the norm is re-computed explicitly.

    x = 1.0 - x * x;

    if ( x < 1.0e-4 || x > 1.0 )
    {
      error  = rscale * vspace.norm2( r );
    }
    else
    {
      error *= std::sqrt ( x );
    }

    store_ ( v, q, k );

    iiter++;
    k++;

    nextIterEvent.emit ( iiter, error, *this );

    testCancelled_ ();

    if ( error <= (0.95 * precision_) || error > MAX_RESIDUAL )
    {
      break;
    }
  }

  z = lhs;

  precon.matmul ( lhs, z );

  if ( vectCount_ >= maxVectors_ )
  {
    print ( System::debug( myName_ ), myName_,
            " : maximum storage reached; "
            "cleared all search vectors\n" );

    vectCount_ = 0;
  }
}


//-----------------------------------------------------------------------
//   reset_
//-----------------------------------------------------------------------


void GCR::reset_ ()
{
  if ( vVectors_.size(1) > 0 )
  {
    Matrix  empty;

    vVectors_.ref ( empty );
    qVectors_.ref ( empty );
  }

  vectCount_ = 0;
}


//-----------------------------------------------------------------------
//   store_
//-----------------------------------------------------------------------


void GCR::store_

  ( const Vector&  v,
    const Vector&  q,
    idx_t          i )

{
  const idx_t  n = vVectors_.size (1);

  if ( n == 0 )
  {
    precon_->resetEvents ();
    matrix_->resetEvents ();
  }

  GMRESUtils::storeVector ( vVectors_, v, i, maxVectors_ );
  GMRESUtils::storeVector ( qVectors_, q, i, maxVectors_ );

  if ( vVectors_.size(1) > n )
  {
    print ( System::debug( myName_ ), myName_,
            " : allocated memory for 2 x ", vVectors_.size(1),
            " search vectors\n" );
  }

  if ( i >= vectCount_ )
  {
    vectCount_ = i + 1;
  }
}


JIVE_END_PACKAGE( solver )
