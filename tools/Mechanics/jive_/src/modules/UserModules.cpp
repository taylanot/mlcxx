/*
 *  UserModules Implementation
 *  
 *  Every module created by user should be declared here inside
 *  declareUserModules function!
 *
 *  Ozgur Taylan Turan, February 2022
 *
 */

//------------------------------------------------------------------------------
//    Base Headers
//------------------------------------------------------------------------------

#include <jive/fem/InputModule.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/LinsolveModule.h>

//------------------------------------------------------------------------------
//    Additional Headers
//------------------------------------------------------------------------------

#include "UserModules.h"

//------------------------------------------------------------------------------
//    declareUserModules
//------------------------------------------------------------------------------

void declareUserModules ()
{
    // Base Modules
    declareInputModule ();
    declareLinsolveModule ();

    // User Modules
    declareTestModule ();  
}


//------------------------------------------------------------------------------
//    declareInputModule
//------------------------------------------------------------------------------


void declareInputModule ()
{
  using jive::app::ModuleFactory;
  using jive::fem::InputModule;

  ModuleFactory::declare ( "input", &InputModule::makeNew );
}

//------------------------------------------------------------------------------
//    declareLinsolveModule
//------------------------------------------------------------------------------


void declareLinsolveModule()
{
  using jive::app::ModuleFactory;
  using jive::implict::LinsolveModule;

  ModuleFactory::declare ( "linsolve", &LinsolveModule::makeNew );
}
