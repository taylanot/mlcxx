
#include <jem/util/Properties.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include "StepModule.h"

//=======================================================================
//   class StepModule
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

StepModule::StepModule ( const String& name ) : Module ( name )
{}

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

Module::Status StepModule::run ( const Properties& globdat )
{
  using jive::model::Model;
  using jive::model::Actions;

  Ref<Model>  model = Model::get ( globdat, getContext() );

  model->takeAction ( Actions::COMMIT, Properties(), globdat );

  return OK;
}
