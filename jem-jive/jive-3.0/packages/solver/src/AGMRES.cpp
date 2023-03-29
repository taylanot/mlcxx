
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
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Event.h>
#include <jive/util/utilities.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/declare.h>
#include <jive/solver/Names.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/algebra/GramSchmidt.h>
#include <jive/solver/SolverInfo.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/Preconditioner.h>
#include <jive/solver/IterativeSolverException.h>
#include <jive/solver/GMRESUtils.h>
#include <jive/solver/GMRES.h>
#include <jive/solver/AGMRES.h>


JEM_DEFINE_CLASS( jive::solver::AGMRES );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::System;
using jive::algebra::newGramSchmidt;


//=======================================================================
//   class AGMRES::SubPrecon_
//=======================================================================


class AGMRES::SubPrecon_ : public Object
{
 public:

  typedef SubPrecon_      Self;
  typedef Object          Super;


                          SubPrecon_

    ( double                alpha,
      const Vector&         kryvec,
      const Vector&         hessmat,
      const Matrix&         rotmat,
      const Matrix&         krylov );

  inline idx_t            size          () const;

  void                    matmul

    ( const Vector&         lhs,
      const Vector&         rhs,
      VectorSpace&          vspace,
      Scratch_&             scratch )      const;


 protected:

  virtual                ~SubPrecon_    ();


 private:

  double                  alpha_;
  Vector                  kryvec_;
  Vector                  hessmat_;
  Matrix                  rotmat_;
  Matrix                  krylov_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


AGMRES::SubPrecon_::SubPrecon_

  ( double         alpha,
    const Vector&  kryvec,
    const Vector&  hessmat,
    const Matrix&  rotmat,
    const Matrix&  krylov ) :

    alpha_   ( alpha ),
    kryvec_  ( kryvec .clone() ),
    hessmat_ ( hessmat.clone() ),
    rotmat_  ( rotmat .clone() ),
    krylov_  ( krylov .clone() )

{}


AGMRES::SubPrecon_::~SubPrecon_ ()
{}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t AGMRES::SubPrecon_::size () const
{
  return krylov_.size (1);
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void AGMRES::SubPrecon_::matmul

  ( const Vector&  lhs,
    const Vector&  rhs,
    VectorSpace&   vspace,
    Scratch_&      scratch ) const

{
  typedef GMRESUtils  Utils;

  using jem::numeric::axpy;
  using jem::numeric::matmul;

  const idx_t  k = krylov_.size (1);

  Vector&      a = scratch.vec1;
  Vector&      b = scratch.vec2;

  double       x;


  a.resize ( k );
  b.resize ( k );

  vspace.project ( a, rhs, krylov_ );

  b = a;

  Utils::rotate  ( b, rotmat_  );
  Utils::backsub ( b, hessmat_ );

  x  = alpha_ * b[k - 1];
  b -= a;

  matmul ( lhs, krylov_, b );
  axpy   ( lhs, -x, kryvec_ );
  axpy   ( lhs, 1.0, rhs );
}


//=======================================================================
//   class AGMRES
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char*  AGMRES::TYPE_NAME    = "AGMRES";
const idx_t  AGMRES::MAX_VECTORS  = 200;
const idx_t  AGMRES::RESTART_ITER = GMRES::RESTART_ITER;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


AGMRES::AGMRES

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

  maxVectors_  = MAX_VECTORS;
  restartIter_ = RESTART_ITER;

  connect ( vspace_->getDofSpace()->newSizeEvent,
            this, & Self::reset_ );
}


AGMRES::~AGMRES ()
{}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void AGMRES::getInfo ( const Properties& info ) const
{
  double  musage   = 0.0;

  idx_t   precSize = psize_ ();
  idx_t   vecCount = precSize + krylov_.size (1);


  Super::getInfo ( info );

  info.find ( musage, SolverInfo::MEM_USAGE );

  musage +=  (double) sizeof(double) *
             (double) vecCount       *
             (double) krylov_.size(0);

  info.set ( SolverInfo::TYPE_NAME, TYPE_NAME );
  info.set ( SolverInfo::MEM_USAGE, musage    );
  info.set ( "vectorCount",         precSize  );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void AGMRES::configure ( const Properties& props )
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
                           0, maxOf( count ) );

    if ( found )
    {
      setMaxVectors ( count );
    }

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


void AGMRES::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super ::getConfig ( props );
  grams_->getConfig ( props );

  myProps.set ( PropNames::RESTART_ITER, restartIter_ );
  myProps.set ( PropNames::MAX_VECTORS,  maxVectors_  );
}


//-----------------------------------------------------------------------
//   setMaxVectors
//-----------------------------------------------------------------------


void AGMRES::setMaxVectors ( idx_t count )
{
  JEM_PRECHECK ( count >= 0 );

  if ( count < maxVectors_ )
  {
    const idx_t  n = preconList_.size ();

    idx_t        i, k;

    for ( i = k = 0; i < n; i++ )
    {
      SubPrecon_*  p = preconList_.getAs<SubPrecon_> ( i );

      if ( k + p->size() > count )
      {
        break;
      }

      k += p->size ();
    }

    if ( i < n )
    {
      preconList_.popBack ( n - i );
    }
  }

  maxVectors_ = count;
}


//-----------------------------------------------------------------------
//   getMaxVectors
//-----------------------------------------------------------------------


idx_t AGMRES::getMaxVectors () const
{
  return maxVectors_;
}


//-----------------------------------------------------------------------
//   setRestartIterCount
//-----------------------------------------------------------------------


void AGMRES::setRestartIterCount ( idx_t count )
{
  JEM_PRECHECK ( count > 0 );

  if ( count < restartIter_ )
  {
    krylov_.ref ( Matrix() );
  }

  restartIter_ = count;
}


//-----------------------------------------------------------------------
//   getRestartIterCount
//-----------------------------------------------------------------------


idx_t AGMRES::getRestartIterCount () const
{
  return restartIter_;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> AGMRES::makeNew

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

  return newInstance<Self> ( name, matrix, vspace,
                             cons, precon, grams );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void AGMRES::declare ()
{
  algebra::declareGSAlgorithms ();

  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   updatePrecon_
//-----------------------------------------------------------------------


void AGMRES::updatePrecon_ ()
{
  Super::updatePrecon_ ();
  preconList_.clear    ();
}


//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------


void AGMRES::solve_

  ( idx_t&         iiter,
    double&        error,
    double         rscale,
    const Vector&  lhs,
    const Vector&  rhs )

{
  JEM_PRECHECK ( error > 0.0 );

  using jem::min;
  using jem::max;
  using jem::equal;
  using jem::Float;
  using jem::numeric::modulus;

  typedef GMRESUtils  Utils;

  const idx_t   dofCount = lhs.size ();
  const idx_t   m        = max ( 1_idx, min( maxVectors_,
                                             restartIter_ ) );

  Matrix&       kry      = krylov_;

  Matrix        buf   ( dofCount, 3 );

  Vector        r     ( buf[0] );
  Vector        v     ( buf[1] );
  Vector        z     ( buf[2] );

  Vector        h     ( (m * (m + 1)) / 2 + 2 );
  Matrix        g     ( 2, m );
  Vector        e     ( m + 1 );
  Vector        a     ( m + 2 );

  double        c, s, x, alpha;

  idx_t         i, j, k;


  lhs    = 0.0;
  alpha  = 0.0;
  r      = rhs;
  v      = r * (rscale / error);
  e      = 0.0;
  e[0]   = error / rscale;
  i = j  = 0;

  while ( i < m && iiter < maxIter_ )
  {
    this   ->store_  ( v, i );
    this   ->premul_ ( z, v );
    matrix_->matmul  ( v, z );

    k = i + 1;

    grams_->orthogonalize ( a[slice(BEGIN,k + 2)], v,
                            kry[slice(BEGIN,k)] );

    h[slice(j,j + k)] = a[slice(BEGIN,k)];
    h[j + k] = alpha  = a[k + 1];

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

        premul_ ( lhs, z );

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

  premul_ ( lhs, z );

  k = psize_ ();

  if ( k + i > maxVectors_ )
  {
    // Clear the preconditioner if the current number of iterations is
    // relatively large.

    if ( preconList_.size() > 0 )
    {
      k = k / (preconList_.size() + 1) + preconList_.size ();
    }

    if ( iiter > k )
    {
      preconList_.clear ();

      print ( System::debug ( myName_ ), myName_,
              " : maximum storage reached; "
              "cleared all preconditioners\n" );
    }
  }
  else if ( i > 1 )
  {
    Ref<SubPrecon_>  p =

      newInstance<SubPrecon_> ( alpha, v,
                                h[slice(BEGIN,j)],
                                g[slice(BEGIN,i - 1)],
                                kry[slice(BEGIN,i)] );

    preconList_.pushBack ( p );

    print ( System::debug( myName_ ), myName_,
            " : added preconditioner with size ", p->size(), '\n' );
  }
}


//-----------------------------------------------------------------------
//   reset_
//-----------------------------------------------------------------------


void AGMRES::reset_ ()
{
  krylov_    .ref   ( Matrix() );
  preconList_.clear ();
}


//-----------------------------------------------------------------------
//   psize_
//-----------------------------------------------------------------------


idx_t AGMRES::psize_ () const
{
  idx_t  k = 0;

  for ( idx_t i = 0; i < preconList_.size(); i++ )
  {
    SubPrecon_*  p = preconList_.getAs<SubPrecon_> ( i );

    k += p->size ();
  }

  return k;
}


//-----------------------------------------------------------------------
//   store_
//-----------------------------------------------------------------------


void AGMRES::store_

  ( const Vector&  v,
    idx_t          i )

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


//-----------------------------------------------------------------------
//   premul_
//-----------------------------------------------------------------------


void AGMRES::premul_

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  if ( preconList_.size() == 0 )
  {
    precon_->matmul ( lhs, rhs );
  }
  else
  {
    Vector&  tmp = scratch_.vec3;

    tmp.resize ( rhs.size() );

    tmp = rhs;

    for ( idx_t i = preconList_.size() - 1; i >= 0; i-- )
    {
      SubPrecon_*  p = preconList_.getAs<SubPrecon_> ( i );

      p->matmul ( lhs, tmp, *vspace_, scratch_ );

      tmp = lhs;
    }

    precon_->matmul ( lhs, tmp );
  }
}


JIVE_END_PACKAGE( solver )
