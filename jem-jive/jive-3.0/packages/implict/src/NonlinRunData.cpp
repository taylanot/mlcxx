
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


#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/mp/Context.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Event.h>
#include <jive/util/Globdat.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/model/Model.h>
#include <jive/model/StateVector.h>
#include <jive/solver/declare.h>
#include <jive/implict/utilities.h>
#include <jive/implict/NonlinRunData.h>


JIVE_BEGIN_PACKAGE( implict )


using jive::util::Globdat;
using jive::model::Actions;
using jive::model::ActionParams;
using jive::model::StateVector;


//=======================================================================
//   class NonlinRunData
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NonlinRunData::NonlinRunData ( const String& ctx ) :

  Super ( ctx )

{
  frozen = false;

  nformat.setScientific     ();
  nformat.setFractionDigits ( 4 );
}


NonlinRunData::~NonlinRunData ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void NonlinRunData::init ( const Properties& globdat )
{
  using jem::util::connect;

  Super::init ( globdat );

  connect ( dofs->newSizeEvent,  this, & Self::dofsChanged_ );
  connect ( dofs->newOrderEvent, this, & Self::dofsChanged_ );

  dofs->resetEvents ();
}


//-----------------------------------------------------------------------
//   initSolver
//-----------------------------------------------------------------------


void NonlinRunData::initSolver

  ( const String&       name,
    double              prec,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  solver::declareSolvers ();

  Properties  params =

    newSolverParams ( globdat, sysmat, precon, dofs );

  model->takeAction ( Actions::GET_SOLVER_PARAMS, params, globdat );

  solver = newSolver ( name, context,
                       conf, props, params, globdat );

  // Set the default solver properties.

  solver->setMode      ( Solver::LENIENT_MODE );
  solver->setPrecision ( 0.5 * prec );

  solver->configure    ( props );
  solver->getConfig    ( conf  );
}


//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------


bool NonlinRunData::commit ( const Properties& globdat )
{
  bool  accept = Super::commit ( globdat );

  if ( accept )
  {
    StateVector::updateOld ( dofs, globdat );
  }

  return accept;
}


//-----------------------------------------------------------------------
//   getBounds
//-----------------------------------------------------------------------


bool NonlinRunData::getBounds

  ( const Vector&      lb,
    const Vector&      ub,
    const Properties&  globdat )

{
  JEM_PRECHECK ( lb.size() == ub.size() );

  bool        result = false;

  Properties  params;

  params.set ( ActionParams::LOWER_BOUND, lb );
  params.set ( ActionParams::UPPER_BOUND, ub );

  if ( model->takeAction( Actions::GET_BOUNDS, params, globdat ) )
  {
    lb     = where ( lb > ub, ub, lb );
    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void NonlinRunData::cancel ( const Properties& globdat )
{
  StateVector::restoreNew ( dofs, globdat );
  Super      ::cancel     ( globdat );
}


//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------


void NonlinRunData::dofsChanged_ ()
{
  if ( frozen )
  {
    dofs->resetEvents ();

    throw jem::IllegalOperationException (
      context,
      "dof space changed during the execution "
      "of the non-linear solver"
    );
  }
}


JIVE_END_PACKAGE( implict )
