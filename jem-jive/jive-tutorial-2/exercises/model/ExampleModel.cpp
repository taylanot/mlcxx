
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
  // Use "props" to initialise the member "parameter_".

  parameter_ = 1.0;
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool ExampleModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  // Continue here ...

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
  // Implement this ...
}

//-----------------------------------------------------------------------
//   declareExampleModel
//-----------------------------------------------------------------------

// Implement the function declareExampleModel here ...

