
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jive/model/ModelFactory.h>
#include "ExampleModel.h"

using namespace jem;

//=======================================================================
//   class ExampleModel
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

ExampleModel::ExampleModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Model ( name )

{
  Properties  myProps = props.findProps ( myName_ );
  Properties  myConf  = conf .makeProps ( myName_ );

  myProps.get ( parameter_, "parameter" );
  myConf .set ( "parameter", parameter_ );
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool ExampleModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  System::out() << "action : "    << action << "\n";
  System::out() << "params :\n\n" << params << "\n\n";

  return false;
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newExampleModel
//-----------------------------------------------------------------------

Ref<Model>           newExampleModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<ExampleModel> ( name, conf, props, globdat );
}

//-----------------------------------------------------------------------
//   declareExampleModel
//-----------------------------------------------------------------------

void declareExampleModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "Example", newExampleModel );
}

