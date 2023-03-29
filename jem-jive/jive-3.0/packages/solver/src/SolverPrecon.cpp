
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


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/solver/Names.h>
#include <jive/solver/Solver.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/PreconFactory.h>
#include <jive/solver/SolverPrecon.h>


JEM_DEFINE_CLASS( jive::solver::SolverPrecon );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;


//=======================================================================
//   class SolverPrecon
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SolverPrecon::TYPE_NAME = "Solver";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SolverPrecon::SolverPrecon

  ( const String&        name,
    const Ref<Solver>&   solver ) :

    Super   ( name   ),
    solver_ ( solver )

{
  JEM_PRECHECK ( solver );

  int  mode = solver_->getMode  ();

  mode |= (Solver::PRECON_MODE | Solver::LENIENT_MODE);

  solver_->setMode ( mode );

  connect_ ();
}


SolverPrecon::~SolverPrecon ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SolverPrecon::resetEvents ()
{
  solver_->getMatrix()->resetEvents ();
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape SolverPrecon::shape () const
{
  return solver_->getMatrix()->shape ();
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SolverPrecon::start ()
{
  solver_->start ();
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SolverPrecon::finish ()
{
  solver_->finish ();
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SolverPrecon::update ()
{}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void SolverPrecon::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  lhs = 0.0;

  solver_->improve ( lhs, rhs );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SolverPrecon::configure ( const Properties& props )
{
  Super  ::configure  ( props );
  solver_->configure  ( props );

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SolverPrecon::getConfig ( const Properties& conf ) const
{
  Super  ::getConfig ( conf );
  solver_->getConfig ( conf );
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool SolverPrecon::hasTrait ( const String& trait ) const
{
  using jive::algebra::MatrixTraits;

  if ( trait == MatrixTraits::SYMMETRIC ||
       trait == MatrixTraits::DISTRIBUTED )
  {
    return solver_->getMatrix()->hasTrait ( trait );
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* SolverPrecon::getConstraints () const
{
  return solver_->getConstraints ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Precon> SolverPrecon::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Ref<Solver>  solver =

    newSolver ( joinNames( name, PropNames::SOLVER ),
                conf, props, params, globdat );

  if ( ! solver )
  {
    return nullptr;
  }
  else
  {
    return newInstance<Self> ( name, solver );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void SolverPrecon::declare ()
{
  PreconFactory::declare ( TYPE_NAME,  & Self::makeNew );
  PreconFactory::declare ( CLASS_NAME, & Self::makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void SolverPrecon::connect_ ()
{
  using jem::util::connect;

  AbstractMatrix*  matrix = solver_->getMatrix ();

  connect ( matrix->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( matrix->newStructEvent, this, & Self::structChanged_ );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void SolverPrecon::valuesChanged_ ()
{
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void SolverPrecon::structChanged_ ()
{
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( solver )
