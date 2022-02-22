 /*
 *  TestModule Implementation
 *
 *  Ozgur Taylan Turan, February 2022
 *
 */

//------------------------------------------------------------------------------
//    Base Headers
//------------------------------------------------------------------------------

#include <jem/base/System.h>
#include <jive/app/ModuleFactory.h>


//------------------------------------------------------------------------------
//    Additional Headers
//------------------------------------------------------------------------------

#include "TestModule.h"

//------------------------------------------------------------------------------
//    Using Declarations
//------------------------------------------------------------------------------

using jem::io::endl;
using jem::util::Properties;
using jive::app::Module;

//==============================================================================
//    class TestModule
//==============================================================================

//------------------------------------------------------------------------------
//    static data
//------------------------------------------------------------------------------


const char* TestModule::TYPE_NAME = "TestModule";


//------------------------------------------------------------------------------
//    constructor & destructor
//------------------------------------------------------------------------------


TestModule::TestModule( const String& name ) : Super( name )
{
  message_ = "TestModule is working!";
}

TestModule::~TestModule()
{}


//------------------------------------------------------------------------------
//    init
//------------------------------------------------------------------------------

Module::Status TestModule::init

  ( const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  System::out() << "Initializing:" << myName_ << endl;
  System::out() << message_ << endl;
  return DONE;
}

//------------------------------------------------------------------------------
//    configure_
//------------------------------------------------------------------------------


void TestModule::configure_

  ( const Properties& conf,
    const Properties& props,
    const Properties& globdat )
{
  Properties  myConf  = conf.makeProps ( myName_ );
  Properties  myProps = props.findProps ( myName_ );
}

//------------------------------------------------------------------------------
//    makeNew
//------------------------------------------------------------------------------

Ref<Module>  TestModule::makeNew

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  return newInstance<Self> ();
}

//------------------------------------------------------------------------------
//    declareTestModule
//------------------------------------------------------------------------------

void declareTestModule()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TestModule::TYPE_NAME, &TestModule::makeNew );
}
