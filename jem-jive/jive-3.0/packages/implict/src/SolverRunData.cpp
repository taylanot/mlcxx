
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


#include <jem/base/Float.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/utilities.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/solver/declare.h>
#include <jive/implict/utilities.h>
#include <jive/implict/SolverRunData.h>


JIVE_BEGIN_PACKAGE( implict )


using jem::Float;
using jem::ArithmeticException;
using jive::util::Globdat;
using jive::model::Actions;
using jive::model::ActionParams;


//=======================================================================
//   class SolverRunData
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SolverRunData::SolverRunData ( const String& ctx ) :

  context ( ctx )

{}


SolverRunData::~SolverRunData ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void SolverRunData::init ( const Properties&  globdat )
{
  Properties  params;

  Globdat::initStep ( globdat );

  model  = Model       :: get ( globdat, context );
  dofs   = DofSpace    :: get ( globdat, context );
  cons   = Constraints :: get ( dofs,    globdat );
  vspace = VectorSpace :: get ( dofs,    globdat );

  params.set ( ActionParams::CONSTRAINTS, cons );

  model->takeAction ( Actions::NEW_MATRIX0, params, globdat );

  if ( ! params.find( sysmat, ActionParams::MATRIX0 ) )
  {
    String  name = model->getName ();

    throw jem::IllegalInputException (
      context,
      String::format (
        "failed to get a matrix "
        "(model `%s\' ignored the action `%s\')",
        name,
        Actions::NEW_MATRIX0
      )
    );
  }

  model->takeAction ( Actions::NEW_PRECON0, params, globdat );
  params.find       ( precon, ActionParams::PRECON0 );
}


//-----------------------------------------------------------------------
//   initSolver
//-----------------------------------------------------------------------


void SolverRunData::initSolver

  ( const String&       name,
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

  solver->configure  ( props );
  solver->getConfig  ( conf  );
}


//-----------------------------------------------------------------------
//   getExtVector
//-----------------------------------------------------------------------


void SolverRunData::getExtVector

  ( const Vector&      fext,
    const Properties&  globdat )

{
  Properties  params;

  fext = 0.0;

  params.set        ( ActionParams::EXT_VECTOR, fext );
  model->takeAction ( Actions::GET_EXT_VECTOR, params, globdat );

  if ( Float::isNaN( sum( fext ) ) )
  {
    throw ArithmeticException (
      context,
      "the external vector contains invalid data: NaN"
    );
  }
}


//-----------------------------------------------------------------------
//   getIntVector
//-----------------------------------------------------------------------


void SolverRunData::getIntVector

  ( const Vector&      fint,
    const Properties&  globdat )

{
  Properties  params;

  fint = 0.0;

  params.set        ( ActionParams::INT_VECTOR, fint );
  model->takeAction ( Actions::GET_INT_VECTOR, params, globdat );

  if ( Float::isNaN( sum( fint ) ) )
  {
    throw ArithmeticException (
      context,
      "the internal vector contains invalid data: NaN"
    );
  }
}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void SolverRunData::updateMatrix

  ( const Vector&      fint,
    const Properties&  globdat )

{
  Properties  params;

  fint = 0.0;

  params.set ( ActionParams::INT_VECTOR, fint );

  model->takeAction ( Actions::UPD_MATRIX0, params, globdat );

  if ( precon )
  {
    model->takeAction ( Actions::UPD_PRECON0, params, globdat );
  }

  if ( Float::isNaN( sum( fint ) ) )
  {
    throw ArithmeticException (
      context,
      "the internal vector contains invalid data: NaN"
    );
  }
}


//-----------------------------------------------------------------------
//   updateConstraints
//-----------------------------------------------------------------------


void SolverRunData::updateConstraints ( const Properties& globdat )
{
  Properties  params;

  params.set        ( ActionParams::CONSTRAINTS, cons );
  model->takeAction ( Actions::GET_CONSTRAINTS, params, globdat );
  cons ->compress   ();
}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


void SolverRunData::advance ( const Properties& globdat )
{
  Globdat::advanceStep ( globdat );
  model  ->takeAction  ( Actions::ADVANCE, Properties(), globdat );
}


//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------


bool SolverRunData::commit ( const Properties& globdat )
{
  Properties  params;

  bool  accept = true;

  model->takeAction ( Actions::CHECK_COMMIT, params, globdat );

  params.find ( accept, ActionParams::ACCEPT );

  if ( accept )
  {
    params.clear ();

    model  ->takeAction ( Actions::COMMIT, params, globdat );
    Globdat::commitStep ( globdat );
  }

  return accept;
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void SolverRunData::cancel ( const Properties& globdat )
{
  Properties  params;

  Globdat::restoreStep ( globdat );
  model  ->takeAction  ( Actions::CANCEL, params, globdat );
}


JIVE_END_PACKAGE( implict )
