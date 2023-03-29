
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

// Continue here ...

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
  // Implement this ...
}

//-----------------------------------------------------------------------
//   declareExampleModel
//-----------------------------------------------------------------------

void declareExampleModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "Example", newExampleModel );
}

