
#include <jive/app/InfoModule.h>
#include <jive/app/InitModule.h>
#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/Application.h>
#include <jive/fem/declare.h>
#include <jive/fem/InputModule.h>
#include <jive/fem/ShapeModule.h>
#include <jive/mesh/MeshgenModule.h>
#include <jive/model/declare.h>
#include <jive/femodel/declare.h>
#include <jive/implict/LinsolveModule.h>
#include <jive/gl/DisplayModule.h>
#include <jive/gl/FemViewModule.h>

using namespace jem;

using jive::app::Module;
using jive::app::InfoModule;
using jive::app::InitModule;
using jive::app::Application;
using jive::app::ChainModule;
using jive::app::ControlModule;
using jive::app::ReportModule;
using jive::fem::InputModule;
using jive::fem::ShapeModule;
using jive::mesh::MeshgenModule;
using jive::implict::LinsolveModule;
using jive::gl::DisplayModule;
using jive::gl::FemViewModule;

//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------

Ref<Module> mainModule ()
{
  Ref<ChainModule>  chain = newInstance<ChainModule> ();
  Ref<Module>       module;

  jive::fem    ::declareMBuilders ();
  jive::model  ::declareModels    ();
  jive::femodel::declareModels    ();

  chain->pushBack ( newInstance<MeshgenModule>  () );
  chain->pushBack ( newInstance<InputModule>    () );
  chain->pushBack ( newInstance<ShapeModule>    () );
  chain->pushBack ( newInstance<InitModule>     () );
  chain->pushBack ( newInstance<InfoModule>     () );
  chain->pushBack ( newInstance<LinsolveModule> () );
  chain->pushBack ( newInstance<FemViewModule>  () );
  chain->pushBack ( newInstance<ControlModule>  () );

  module = newInstance<ReportModule>  ( "report", chain );
  module = newInstance<DisplayModule> ( module );

  return module;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ( int argc, char** argv )
{
  return Application::exec ( argc, argv, & mainModule );
}
