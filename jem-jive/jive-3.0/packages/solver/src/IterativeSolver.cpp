
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
#include <jive/solver/IterativeSolver.h>


JEM_DEFINE_CLASS( jive::solver::IterativeSolver );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class IterativeSolver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const idx_t  IterativeSolver::MAX_ITER = 2000;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


IterativeSolver::IterativeSolver ( const String& name ) :

  Super ( name )

{}


IterativeSolver::~IterativeSolver ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void IterativeSolver::configure ( const Properties& props )
{
  using jem::maxOf;

  Super::configure ( props );

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    idx_t       maxIter;

    if ( myProps.find( maxIter, PropNames::MAX_ITER,
                       1, maxOf( maxIter ) ) )
    {
      setMaxIterCount ( maxIter );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void IterativeSolver::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super::getConfig ( props );

  myProps.set ( PropNames::MAX_ITER, getMaxIterCount() );
}


JIVE_END_PACKAGE( solver )
