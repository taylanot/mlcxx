/*
 *  StressModel Implementation
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


#include "StressModel.h"


using namespace jem;

//==============================================================================
//    class StressModel
//==============================================================================


//------------------------------------------------------------------------------
//    static data
//------------------------------------------------------------------------------


const char* StressModel::TYPE_NAME = "StressModel";


//------------------------------------------------------------------------------
//    constructor
//------------------------------------------------------------------------------

StressModel::StressModel
  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat ) :

    Model ( name )

{
  Properties  myProps = props.findProps ( myName_ );
  Properties  myConf  = conf .makeProps ( myName_ );

}

//------------------------------------------------------------------------------
//    takeAction
//------------------------------------------------------------------------------

bool StressModel::takeAction
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

Ref<Model>  StressModel::makeNew
  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )
{
  return newInstance<StressModel> ( name, conf, props, globdat );
}

//------------------------------------------------------------------------------
//    declareStressModel
//------------------------------------------------------------------------------

void declareStressModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( StressModel::TYPE_NAME, StressModel::makeNew );
}

