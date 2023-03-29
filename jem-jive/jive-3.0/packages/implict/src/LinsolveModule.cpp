
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


#include <jem/base/System.h>
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/solver/Solver.h>
#include <jive/model/StateVector.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/Names.h>
#include <jive/implict/SolverInfo.h>
#include <jive/implict/SolverRunData.h>
#include <jive/implict/LinsolveModule.h>


JEM_DEFINE_CLASS( jive::implict::LinsolveModule );


JIVE_BEGIN_PACKAGE( implict )


using jem::newInstance;


//=======================================================================
//   class LinsolveModule::RunData_
//=======================================================================


class LinsolveModule::RunData_ : public SolverRunData
{
 public:

  typedef SolverRunData   Super;
  typedef RunData_        Self;


  explicit inline         RunData_

    ( const String&         ctx );


 public:

  Matrix                  mbuf;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline LinsolveModule::RunData_::RunData_ ( const String& ctx ) :

  Super ( ctx )

{}


//=======================================================================
//   class LinsolveModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  LinsolveModule::TYPE_NAME = "Linsolve";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LinsolveModule::LinsolveModule ( const String& name ) :

  Super ( name )

{}


LinsolveModule::~LinsolveModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status LinsolveModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  rundat_ = nullptr;

  Ref<RunData_>  newdat = newInstance<RunData_> ( getContext() );
  String         name   = joinNames ( myName_, PropNames::SOLVER );

  newdat->init       ( globdat );
  newdat->initSolver ( name, conf, props, globdat );

  // Everything OK, so commit the changes

  rundat_.swap ( newdat );

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void LinsolveModule::shutdown ( const Properties& globdat )
{
  rundat_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LinsolveModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( rundat_ )
  {
    rundat_->solver->configure ( props );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LinsolveModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  if ( rundat_ )
  {
    rundat_->solver->getConfig ( conf );
  }
}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


void LinsolveModule::advance ( const Properties& globdat )
{
  if ( ! rundat_ )
  {
    notAliveError ( JEM_FUNC );
  }

  rundat_->advance ( globdat );
}


//-----------------------------------------------------------------------
//   solve
//-----------------------------------------------------------------------


void LinsolveModule::solve

  ( const Properties&  info,
    const Properties&  globdat )

{
  using jem::System;
  using jem::Exception;
  using jive::util::Globdat;
  using jive::model::StateVector;

  if ( ! rundat_ )
  {
    notAliveError ( JEM_FUNC );
  }

  RunData_&    d        = * rundat_;

  const idx_t  dofCount = d.dofs->dofCount ();

  Vector       a, b, u;


  if ( dofCount != d.mbuf.size(0) || d.mbuf.size(1) != 2 )
  {
    d.mbuf.resize ( dofCount, 2 );
  }

  a.ref ( d.mbuf[0] );
  b.ref ( d.mbuf[1] );

  StateVector::get ( u, d.dofs, globdat );

  // Save the state vector, set it to zero and assemble the system of
  // equations. Setting the state vector to zero is necessary to
  // obtain the correct internal force vector (which is stored in the
  // vector a).

  b = u;
  a = 0.0;
  u = 0.0;

  try
  {
    print ( System::info( myName_ ),
            "Assembling the global system of equations ...\n" );

    d.updateConstraints (    globdat );
    d.updateMatrix      ( a, globdat );
  }
  catch ( const Exception& )
  {
    u = b;

    throw;
  }

  // Restore the state vector and compute the right-hand side vector.

  u = b;

  d.getExtVector ( b, globdat );

  b -= a;

  print ( System::info( myName_ ),
          "Solving the global system of equations ...\n\n" );

  d.solver->improve ( u, b );
  d.updateModel     ( globdat );

  info.set ( SolverInfo::CONVERGED,  true );
  info.set ( SolverInfo::ITER_COUNT, 1    );

  // Store solver info.

  if ( Globdat::hasVariable( myName_, globdat ) )
  {
    d.solver->getInfo ( Globdat::getVariables( myName_, globdat ) );
  }
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void LinsolveModule::cancel ( const Properties& globdat )
{
  if ( ! rundat_ )
  {
    notAliveError ( JEM_FUNC );
  }

  rundat_->cancel ( globdat );
}


//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------


bool LinsolveModule::commit ( const Properties& globdat )
{
  if ( ! rundat_ )
  {
    notAliveError ( JEM_FUNC );
  }

  return rundat_->commit ( globdat );
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void LinsolveModule::setPrecision ( double eps )
{
  if ( rundat_ )
  {
    return setPrecision ( eps );
  }
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double LinsolveModule::getPrecision () const
{
  if ( rundat_ )
  {
    return getPrecision ();
  }
  else
  {
    return 0.0;
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> LinsolveModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void LinsolveModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( implict )
