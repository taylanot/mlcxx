
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


#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/solver/Names.h>
#include <jive/solver/EigenSolver.h>


JEM_DEFINE_CLASS( jive::solver::EigenSolver );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class EigenSolver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  EigenSolver::PRECISION = 1.0e-3;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


EigenSolver::EigenSolver ( const String& name ) :

  Super ( name )

{}


EigenSolver::~EigenSolver ()
{}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void EigenSolver::getInfo ( const Properties& info ) const
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void EigenSolver::configure ( const Properties& props )
{
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    double      eps;

    if ( myProps.find( eps, PropNames::PRECISION,
                       0.0, maxOf( eps ) ) )
    {
      setPrecision ( eps );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void EigenSolver::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::PRECISION, getPrecision() );
}


//-----------------------------------------------------------------------
//   getRealEigenValues
//-----------------------------------------------------------------------


void EigenSolver::getRealEigenValues

  ( const Vector&  evals,
    idx_t          smallCount )

{
  const idx_t  n = evals.size();

  Matrix       tmpVals ( 2, n );

  getEigenValues  ( tmpVals, smallCount );

  evals = tmpVals(0,ALL);
}


//-----------------------------------------------------------------------
//   getRealEigenVectors
//-----------------------------------------------------------------------


void EigenSolver::getRealEigenVectors

  ( const Vector&  evals,
    const Matrix&  evecs,
    idx_t            smallCount )

{
  JEM_PRECHECK ( evals.size() == evecs.size(1) );

  const idx_t  m = evecs.size(0);
  const idx_t  n = evecs.size(1);

  Matrix       tmpVals ( 2, n );
  Cubix        tmpVecs ( 2, m, n );


  getEigenVectors ( tmpVals, tmpVecs, smallCount );

  evals = tmpVals(0,ALL);
  evecs = tmpVecs(0,ALL,ALL);
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String EigenSolver::getContext () const
{
  return NamedObject::makeContext ( "eigen solver", myName_ );
}


JIVE_END_PACKAGE( solver )
