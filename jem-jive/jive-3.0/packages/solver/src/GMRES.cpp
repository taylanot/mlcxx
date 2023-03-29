
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
#include <jem/base/System.h>
#include <jem/base/Float.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/numeric/utilities.h>
#include <jem/util/Event.h>
#include <jive/util/utilities.h>
#include <jive/util/DofSpace.h>
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
#include <jive/solver/GMRES.h>


JEM_DEFINE_CLASS( jive::solver::GMRES );


JIVE_BEGIN_PACKAGE( solver )


using jive::algebra::newGramSchmidt;


//=======================================================================
//   class GMRES
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GMRES::TYPE_NAME    = "GMRES";
const idx_t  GMRES::RESTART_ITER = 100;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GMRES::GMRES

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<VectorSpace>     vspace,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  precon,
    Ref<GramSchmidt>     grams ) :

    Super  ( name, matrix, vspace, cons, precon ),
    grams_ ( grams  )

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

  restartIter_ = RESTART_ITER;

  connect ( vspace_->getDofSpace()->newSizeEvent,
            this, & Self::reset_ );
}


GMRES::~GMRES ()
{}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void GMRES::getInfo ( const Properties& info ) const
{
  double  musage = 0.0;


  Super::getInfo ( info );

  info.find ( musage, SolverInfo::MEM_USAGE );

  musage += (double) sizeof(double)  *
            (double) krylov_.size(0) *
            (double) krylov_.size(1);

  info.set ( SolverInfo::TYPE_NAME, TYPE_NAME );
  info.set ( SolverInfo::MEM_USAGE, musage );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void GMRES::configure ( const Properties& props )
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
                           PropNames::RESTART_ITER,
                           1, maxOf( count ) );

    if ( found )
    {
      setRestartIterCount ( count );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void GMRES::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super ::getConfig ( props );
  grams_->getConfig ( props );

  myProps.set ( PropNames::RESTART_ITER, restartIter_ );
}


//-----------------------------------------------------------------------
//   setRestartIterCount
//-----------------------------------------------------------------------


void GMRES::setRestartIterCount ( idx_t count )
{
  JEM_PRECHECK ( count > 0 );

  if ( count < restartIter_ )
  {
    reset_ ();
  }

  restartIter_ = count;
}


//-----------------------------------------------------------------------
//   getRestartIterCount
//-----------------------------------------------------------------------


idx_t GMRES::getRestartIterCount () const
{
  return restartIter_;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> GMRES::makeNew

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


void GMRES::declare ()
{
  algebra::declareGSAlgorithms ();

  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------


void GMRES::solve_

  ( idx_t&         iiter,
    double&        error,
    double         rscale,
    const Vector&  lhs,
    const Vector&  rhs )

{
  JEM_PRECHECK ( error > 0.0 );

  using jem::equal;
  using jem::Float;
  using jem::numeric::modulus;

  typedef GMRESUtils     Utils;

  const AbstractMatrix&  matrix = * matrix_;
  AbstractMatrix&        precon = * precon_;

  const idx_t   dofCount = lhs.size ();
  const idx_t   m        = restartIter_;

  Matrix&       kry      = krylov_;

  Matrix        rvz ( dofCount, 3 );

  Vector        r   ( rvz[0] );
  Vector        v   ( rvz[1] );
  Vector        z   ( rvz[2] );

  Vector        h   ( (m * (m + 1)) / 2 + 2 );
  Matrix        g   ( 2, m );
  Vector        e   ( m + 1 );
  Vector        a   ( m + 2 );

  double        c, s, x;

  idx_t         i, j, k;


  lhs    = 0.0;
  r      = rhs;
  v      = r * (rscale / error);
  e      = 0.0;
  e[0]   = error / rscale;
  i = j  = 0;

  while ( i < m && iiter < maxIter_ )
  {
    store_        ( v, i );
    precon.matmul ( z, v );
    matrix.matmul ( v, z );

    k = i + 1;

    grams_->orthogonalize ( a[slice(BEGIN,k + 2)], v,
                            kry[slice(BEGIN,k)] );

    h[slice(j,j + k)] = a[slice(BEGIN,k)];
    h[j + k]          = a[k + 1];

    // Apply the accumulated Givens rotations to the i-th column of h

    Utils::rotate ( h[slice(j,j + k)], g[slice(BEGIN,i)] );

    // Compute the coefficients for the next Givens rotation

    x      =  1.0 / modulus ( h[j + i], h[j + k] );

    c      =  h[j + i] * x;
    s      = -h[j + k] * x;
    g(0,i) = c;
    g(1,i) = s;

    // Apply the new rotation to the i-th column of h
    // and to the vector e.

    Utils::rotate ( h[j + i], h[j + k], c, s );
    Utils::rotate ( e[i],     e[i + 1], c, s );

    // Continue with the next iteration

    iiter++;
    i  = k;
    j += k;

    // Update the norm of the preconditioned residual and
    // check convergence

    error = rscale * std::fabs ( e[i] );

    nextIterEvent.emit ( iiter, error, *this );

    testCancelled_ ();

    if ( error <= (0.95 * precision_) || error > MAX_RESIDUAL )
    {
      break;
    }

    // Normalize the next Krylov vector.

    if ( i < m )
    {
      // Check for a zero Krylov vector.

      if ( 0.1 * a[i + 1] <= Float::EPSILON * a[i] )
      {
        // Compute the best solution so far ...

        Utils::getSolution ( z,
                             h[slice(BEGIN,j)],
                             e[slice(BEGIN,i)],
                             kry[slice(BEGIN,i)] );

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

      v *= (1.0 / a[i + 1]);
    }
  }

  Utils::getSolution ( z,
                       h[slice(BEGIN,j)],
                       e[slice(BEGIN,i)],
                       kry[slice(BEGIN,i)] );

  precon.matmul ( lhs, z );
}


//-----------------------------------------------------------------------
//   reset_
//-----------------------------------------------------------------------


void GMRES::reset_ ()
{
  krylov_.ref ( Matrix() );
}


//-----------------------------------------------------------------------
//   store_
//-----------------------------------------------------------------------


void GMRES::store_

  ( const Vector&  v,
    idx_t            i )

{
  const idx_t  n = krylov_.size (1);

  if ( n == 0 )
  {
    vspace_->getDofSpace()->resetEvents ();
  }

  GMRESUtils::storeVector ( krylov_, v, i, restartIter_ );

  if ( krylov_.size(1) > n )
  {
    print ( jem::System::debug( myName_ ), myName_,
            " : allocated memory for ", krylov_.size(1),
            " Krylov vectors\n" );
  }
}


JIVE_END_PACKAGE( solver )
