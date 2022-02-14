/*
 *  TestModel Implementation
 *  
 *  Ozgur Taylan Turan, February 2022
 *
 */

//------------------------------------------------------------------------------
//    Base Headers
//------------------------------------------------------------------------------


#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jive/model/ModelFactory.h>

//------------------------------------------------------------------------------
//    Additional Headers
//------------------------------------------------------------------------------


#include "TestModel.h"


using namespace jem;

//==============================================================================
//    class TestModel
//==============================================================================


//------------------------------------------------------------------------------
//    static data
//------------------------------------------------------------------------------


const char* TestModel::TYPE_NAME = "TestModel";


//------------------------------------------------------------------------------
//    constructor
//------------------------------------------------------------------------------

TestModel::TestModel
  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat ) :

    Model ( name )

{
  Properties  myProps = props.findProps ( myName_ );
  Properties  myConf  = conf .makeProps ( myName_ );

  myProps.get ( parameter_, "parameter" );
  System::out() << "props:\n\n" << myProps << "\n\n";
  myConf .set ( "parameter", parameter_ );
  System::out() << "parameter :\n\n" << parameter_ << "\n\n";
}

//------------------------------------------------------------------------------
//    takeAction
//------------------------------------------------------------------------------

bool TestModel::takeAction
  ( const String&     action,
    const Properties& params,
    const Properties& globdat )

{
  System::out() << "action : "    << action << "\n";

  return false;
}

//==============================================================================
//    related functions
//==============================================================================

//------------------------------------------------------------------------------
//    makeNew
//------------------------------------------------------------------------------

Ref<Model>  TestModel::makeNew
  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )
{
  return newInstance<TestModel> ( name, conf, props, globdat );
}

//------------------------------------------------------------------------------
//    declareTestModel
//------------------------------------------------------------------------------

void declareTestModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( TestModel::TYPE_NAME, TestModel::makeNew );
}

