
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/numeric/algebra/Cholesky.h>
#include <jem/numeric/algebra/EigenUtils.h>
#include <jem/util/Properties.h>
#include <jive/util/error.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/solver/SolverInfo.h>
#include <jive/solver/EigenSolverParams.h>
#include <jive/solver/EigenSolverFactory.h>
#include <jive/solver/SymdirEigenSolver.h>


JEM_DEFINE_CLASS( jive::solver::SymdirEigenSolver );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::numeric::Cholesky;
using jem::numeric::EigenUtils;
using jive::algebra::SparseMatrixExt;


//=======================================================================
//   class SymdirEigenSolver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SymdirEigenSolver::TYPE_NAME = "Symdir";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SymdirEigenSolver::SymdirEigenSolver

  ( const String&        name,
    Ref<AbstractMatrix>  lmat,
    Ref<AbstractMatrix>  rmat ) :

    Super     ( name ),
    lhMatrix_ ( lmat ),
    rhMatrix_ ( rmat )

{
  JEM_PRECHECK ( lmat             &&
                 lmat->isSquare() &&
                 lmat->hasExtension<SparseMatrixExt>() );

  if ( rmat )
  {
    JEM_PRECHECK ( rmat->isSquare() &&
                   rmat->hasExtension<SparseMatrixExt>() );
  }

  precision_ = PRECISION;
}


SymdirEigenSolver::~SymdirEigenSolver ()
{}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void SymdirEigenSolver::getInfo ( const Properties& info ) const
{
  info.set ( SolverInfo::TYPE_NAME, TYPE_NAME );
}


//-----------------------------------------------------------------------
//   getEigenValues
//-----------------------------------------------------------------------


void SymdirEigenSolver::getEigenValues

  ( const Matrix&  evals,
    idx_t          smallCount )

{
  JEM_PRECHECK ( evals.size(0) == 2 );

  evals(1,ALL) = 0.0;

  Self::getRealEigenValues ( evals(0,ALL), smallCount );
}


//-----------------------------------------------------------------------
//   getEigenVectors
//-----------------------------------------------------------------------


void SymdirEigenSolver::getEigenVectors

  ( const Matrix&  evals,
    const Cubix&   evecs,
    idx_t          smallCount )

{
  JEM_PRECHECK ( evals.size(0) == 2 &&
                 evecs.size(0) == 2 &&
                 evals.size(1) == evecs.size(2) );

  evals(1,ALL)     = 0.0;
  evecs(1,ALL,ALL) = 0.0;

  Self::getRealEigenVectors ( evals(0,ALL),
                              evecs(0,ALL,ALL), smallCount );
}


//-----------------------------------------------------------------------
//   getRealEigenValues
//-----------------------------------------------------------------------


void SymdirEigenSolver::getRealEigenValues

  ( const Vector&  evals,
    idx_t          smallCount )

{
  JEM_PRECHECK ( evals.size() >= smallCount );

  Matrix       lmat    = toDenseMatrix_ ( *lhMatrix_ );

  const idx_t  evCount = evals.size ();
  const idx_t  msize   = lmat .size (0);

  IdxVector    iperm   ( msize );
  Vector       e       ( msize );

  idx_t        n;


  if ( ! rhMatrix_ )
  {
    EigenUtils::symSolve ( e, lmat );
  }
  else
  {
    Matrix  rmat = toDenseMatrix_ ( *rhMatrix_ );

    getEigenValues_ ( e, lmat, rmat );
  }

  iperm = jem::iarray ( msize );

  jem::sort ( iperm, e );

  n     = jem::min ( msize, smallCount );

  evals = 0.0;

  for ( idx_t i = 0; i < n; i++ )
  {
    evals[i] = e[iperm[i]];
  }

  n = jem::min ( msize   - smallCount,
                 evCount - smallCount );

  for ( idx_t i = 1; i <= n; i++ )
  {
    evals[evCount - i] = e[iperm[msize - i]];
  }
}


//-----------------------------------------------------------------------
//   getRealEigenVectors
//-----------------------------------------------------------------------


void SymdirEigenSolver::getRealEigenVectors

  ( const Vector&  evals,
    const Matrix&  evecs,
    idx_t          smallCount )

{
  JEM_PRECHECK ( evals.size() == evecs.size(1) &&
                 evals.size() >= smallCount );

  Matrix       lmat    = toDenseMatrix_ ( *lhMatrix_ );

  const idx_t  evCount = evals.size ();
  const idx_t  msize   = lmat .size (0);

  IdxVector    jperm   ( msize );
  Matrix       v       ( msize, msize );
  Vector       e       ( msize );

  idx_t        n;


  if ( evecs.size(0) != msize )
  {
    util::shapeError ( getContext  (), "eigenvalue matrix",
                       evecs.shape (),
                       jem::shape  ( msize, evCount ) );
  }

  if ( ! rhMatrix_ )
  {
    EigenUtils::symSolve ( e, v, lmat );
  }
  else
  {
    Matrix  rmat = toDenseMatrix_ ( *rhMatrix_ );

    getEigenVectors_ ( e, v, lmat, rmat );
  }

  jperm = jem::iarray ( msize );

  jem::sort ( jperm, e );

  n     = jem::min ( msize, smallCount );

  evals = 0.0;
  evecs = 0.0;

  for ( idx_t  j = 0; j < n; j++ )
  {
    idx_t  k = jperm[j];

    evals[j] = e[k];

    for ( idx_t i = 0; i < msize; i++ )
    {
      evecs(i,j) = v(i,k);
    }
  }

  n = jem::min ( msize   - smallCount,
                 evCount - smallCount );

  for ( idx_t j = 1; j <= n; j++ )
  {
    idx_t  k = jperm[msize - j];

    evals[evCount - j] = e[k];

    for ( idx_t i = 0; i < msize; i++ )
    {
      evecs(i,evCount - j) = v(i,k);
    }
  }
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void SymdirEigenSolver::setPrecision ( double eps )
{
  JEM_PRECHECK ( eps > 0.0 );

  precision_ = eps;
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double SymdirEigenSolver::getPrecision () const
{
  return precision_;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<EigenSolver> SymdirEigenSolver::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params )

{
  Ref<AbstractMatrix>  lmat;
  Ref<AbstractMatrix>  rmat;


  params.find ( lmat, EigenSolverParams::LEFT_MATRIX  );
  params.find ( rmat, EigenSolverParams::RIGHT_MATRIX );

  if ( ! lmat || ! lmat->hasExtension<SparseMatrixExt>() )
  {
    return nullptr;
  }

  if ( rmat && ! rmat->hasExtension<SparseMatrixExt>() )
  {
    return nullptr;
  }

  return newInstance<Self> ( name, lmat, rmat );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void SymdirEigenSolver::declare ()
{
  EigenSolverFactory::declare ( TYPE_NAME,  & makeNew );
  EigenSolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   getEigenValues_
//-----------------------------------------------------------------------


void SymdirEigenSolver::getEigenValues_

  ( const Vector&  e,
    const Matrix&  lmat,
    const Matrix&  rmat ) const

{
  checkShapes_ ( lmat, rmat );

  if ( ! Cholesky::factor( rmat ) )
  {
    choleskyError_ ();
  }

  Cholesky  ::fsub     ( lmat, rmat );
  Cholesky  ::fsubt    ( lmat, rmat );
  EigenUtils::symSolve ( e,    lmat );
}


//-----------------------------------------------------------------------
//   getEigenVectors_
//-----------------------------------------------------------------------


void SymdirEigenSolver::getEigenVectors_

  ( const Vector&  e,
    const Matrix&  v,
    const Matrix&  lmat,
    const Matrix&  rmat ) const

{
  using jem::dot;

  const idx_t  n = lmat.size (0);


  checkShapes_ ( lmat, rmat );

  if ( ! Cholesky::factor( rmat ) )
  {
    choleskyError_ ();
  }

  Cholesky  ::fsub     ( lmat, rmat );
  Cholesky  ::fsubt    ( lmat, rmat );
  EigenUtils::symSolve ( e, v, lmat );
  Cholesky  ::bsub     ( v,    rmat );

  // Re-normalize the eigenvectors

  for ( idx_t j = 0; j < n; j++ )
  {
    Vector  vj ( v[j] );

    double  x  = std::sqrt ( dot( vj, vj ) );

    if ( ! jem::isTiny( x ) )
    {
      vj *= (1.0 / x);
    }
  }
}


//-----------------------------------------------------------------------
//   toDenseMatrix_
//-----------------------------------------------------------------------


Matrix SymdirEigenSolver::toDenseMatrix_

  ( AbstractMatrix&  amat ) const

{
  using jem::System;

  SparseMatrixExt*  ext  =

    amat.getExtension<SparseMatrixExt> ();

  SparseMatrix  smat       = ext->toSparseMatrix   ();
  IdxVector     rowOffsets = smat.getRowOffsets    ();
  IdxVector     colIndices = smat.getColumnIndices ();
  Vector        values     = smat.getValues        ();

  const idx_t   msize      = smat.size (0);

  Matrix        dmat       ( smat.shape() );


  if ( dmat.size(0) != dmat.size(1) )
  {
    String  name = getMatrixName_ ( amat );

    throw jem::IllegalArgumentException (
      getContext     (),
      String::format (
        "%s should be square but has shape (%d x %d)",
        name,
        dmat.size(0), dmat.size(1)
      )
    );
  }

  if ( ! amat.isSymmetric() )
  {
    print ( System::warn(), myName_, " : non-symmetric ",
            getMatrixName_( amat ), '\n' );
  }

  dmat = 0.0;

  for ( idx_t irow = 0; irow < msize; irow++ )
  {
    idx_t  i = rowOffsets[irow];
    idx_t  n = rowOffsets[irow + 1];

    for ( ; i < n; i++ )
    {
      dmat(irow,colIndices[i]) = values[i];
    }
  }

  return dmat;
}


//-----------------------------------------------------------------------
//   getMatrixName_
//-----------------------------------------------------------------------


String SymdirEigenSolver::getMatrixName_

  ( AbstractMatrix&  amat ) const

{
  if      ( &amat == lhMatrix_ )
  {
    return "left-hand matrix";
  }
  else if ( &amat == rhMatrix_ )
  {
    return "right-hand matrix";
  }
  else
  {
    return "matrix";
  }
}


//-----------------------------------------------------------------------
//   checkShapes_
//-----------------------------------------------------------------------


inline void SymdirEigenSolver::checkShapes_

  ( const Matrix&  lmat,
    const Matrix&  rmat ) const

{
  if ( lmat.size(0) != rmat.size(0) ||
       lmat.size(1) != rmat.size(1) )
  {
    throw jem::IllegalArgumentException (
      getContext (),
      "left-hand matrix and right-hand matrix have different shapes"
    );
  }
}


//-----------------------------------------------------------------------
//   choleskyError_
//-----------------------------------------------------------------------


void SymdirEigenSolver::choleskyError_ () const
{
  throw jem::ArithmeticException (
    getContext (),
    "right-hand matrix is not symmetric positive definite "
    "(Cholesky factorization failed)"
  );
}


JIVE_END_PACKAGE( solver )
