
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


#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jive/util/Globdat.h>
#include <jive/model/Model.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/TransientSolver.h>


JEM_DEFINE_CLASS( jive::implict::TransientSolver );


JIVE_BEGIN_PACKAGE( implict )


//=======================================================================
//   class TransientSolver
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


TransientSolver::TransientSolver

  ( const String&      name,
    Ref<SolverModule>  solver ) :

    Super   ( name ),
    solver_ ( solver )

{
  JEM_PRECHECK ( solver );
}


TransientSolver::~TransientSolver ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status TransientSolver::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::Exception;
  using jive::util::Globdat;

  Ref<Model>  tmodel;
  Ref<Model>  model0;


  Globdat::initTime ( globdat );

  // Wrap the current model in a transient model.

  tmodel_ = nullptr;
  model0  = Model::find    ( globdat );
  tmodel  = newTransModel_ ( conf, props, globdat );

  // Store the transient model in globdat so that the solver can pick
  // it up.

  tmodel->configure ( props, globdat );
  tmodel->getConfig ( conf,  globdat );
  tmodel->store     (        globdat );

  try
  {
    solver_->init ( conf, props, globdat );
  }
  catch ( const jem::Exception& )
  {
    if ( model0 )
    {
      model0->store ( globdat );
    }

    throw;
  }

  // Put back the original model and commit the new transient model.

  if ( model0 )
  {
    model0->store ( globdat );
  }

  tmodel_.swap ( tmodel );

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status TransientSolver::run ( const Properties& globdat )
{
  return solver_->run ( globdat );
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void TransientSolver::shutdown ( const Properties& globdat )
{
  solver_->shutdown ( globdat );

  tmodel_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void TransientSolver::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  solver_->configure ( props, globdat );

  if ( tmodel_ )
  {
    tmodel_->configure ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void TransientSolver::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  solver_->getConfig ( conf, globdat );

  if ( tmodel_ )
  {
    tmodel_->getConfig ( conf, globdat );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool TransientSolver::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  return solver_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


void TransientSolver::advance ( const Properties& globdat )
{
  solver_->advance ( globdat );
}


//-----------------------------------------------------------------------
//   solve
//-----------------------------------------------------------------------


void TransientSolver::solve

  ( const Properties&  info,
    const Properties&  globdat )

{
  solver_->solve ( info, globdat );
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void TransientSolver::cancel ( const Properties& globdat )
{
  solver_->cancel ( globdat );
}


//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------


bool TransientSolver::commit ( const Properties& globdat )
{
  return solver_->commit ( globdat );
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void TransientSolver::setPrecision ( double eps )
{
  solver_->setPrecision ( eps );
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double TransientSolver::getPrecision () const
{
  return solver_->getPrecision ();
}


JIVE_END_PACKAGE( implict )
