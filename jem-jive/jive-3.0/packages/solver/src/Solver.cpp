
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
#include <jive/solver/Solver.h>


JEM_DEFINE_CLASS( jive::solver::Solver );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class Solver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  Solver::PRECISION     = 1.0e-5;
const int     Solver::LENIENT_MODE  = 1 << 0;
const int     Solver::PRECON_MODE   = 1 << 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Solver::Solver ( const String& name ) :

  Super ( name )

{}


Solver::~Solver ()
{}


//-----------------------------------------------------------------------
//   solve
//-----------------------------------------------------------------------


void Solver::solve

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  lhs = 0.0;

  improve ( lhs, rhs );
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void Solver::getInfo ( const Properties& info ) const
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Solver::configure ( const Properties& props )
{
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    double      eps;
    bool        lenient;

    if ( myProps.find( lenient, PropNames::LENIENT ) )
    {
      int  mode = getMode ();

      if ( lenient )
      {
        mode |=  LENIENT_MODE;
      }
      else
      {
        mode &= ~LENIENT_MODE;
      }

      setMode ( mode );
    }

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


void Solver::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );
  bool        lenient = (getMode() & LENIENT_MODE );

  myProps.set ( PropNames::LENIENT,   lenient        );
  myProps.set ( PropNames::PRECISION, getPrecision() );
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* Solver::getConstraints () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Solver::getContext () const
{
  return NamedObject::makeContext ( "solver", myName_ );
}


//=======================================================================
//   class SolverScope
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SolverScope::SolverScope ()
{}


SolverScope::SolverScope ( Solver& solver )
{
  solver.start ();

  solver_ = &solver;
}


SolverScope::~SolverScope ()
{
  if ( solver_ )
  {
    solver_->finish ();

    solver_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SolverScope::start ( Solver& solver )
{
  if ( solver_.get() != &solver )
  {
    if ( solver_ )
    {
      solver_->finish ();
    }

    solver.start ();

    solver_ = &solver;
  }
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SolverScope::finish ()
{
  if ( solver_ )
  {
    solver_->finish ();

    solver_ = nullptr;
  }
}


JIVE_END_PACKAGE( solver )
