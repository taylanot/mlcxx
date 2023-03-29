
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
#include <cfloat>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/solver/Names.h>
#include <jive/solver/DirectSolver.h>


JEM_DEFINE_CLASS( jive::solver::DirectSolver );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class DirectSolver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  DirectSolver::ZERO_THRESHOLD =

  std::sqrt ( DBL_EPSILON );


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DirectSolver::DirectSolver ( const String& name ) :

  Super ( name )

{}


DirectSolver::~DirectSolver ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void DirectSolver::configure ( const Properties& props )
{
  using jem::maxOf;

  Super::configure ( props );

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    bool        found;
    double      threshold;
    idx_t       maxCount;


    found = myProps.find ( threshold,
                           PropNames::ZERO_THRESHOLD,
                           0.0, maxOf ( threshold ) );

    if ( found )
    {
      setZeroThreshold ( threshold );
    }

    found = myProps.find ( maxCount,
                           PropNames::MAX_ZERO_PIVOTS );

    if ( found )
    {
      setMaxZeroPivots ( maxCount );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void DirectSolver::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super::getConfig ( props );

  myProps.set ( PropNames::ZERO_THRESHOLD,  getZeroThreshold() );
  myProps.set ( PropNames::MAX_ZERO_PIVOTS, getMaxZeroPivots() );
}


JIVE_END_PACKAGE( solver )
