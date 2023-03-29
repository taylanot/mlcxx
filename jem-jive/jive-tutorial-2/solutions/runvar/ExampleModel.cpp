
#include <cmath>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include "ExampleModel.h"

using namespace jem;

//=======================================================================
//   class ExampleModel
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

ExampleModel::ExampleModel ( const String& name ) :

  Model ( name )

{}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool ExampleModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;

  if ( action == Actions::COMMIT )
  {
    double  v = std::cos ( 0.2 * (double) iiter );

    iiter++;

    globdat.set ( "var.v", v );

    return true;
  }

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
  return newInstance<ExampleModel> ( name );
}

//-----------------------------------------------------------------------
//   declareExampleModel
//-----------------------------------------------------------------------

void declareExampleModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "Example", newExampleModel );
}

