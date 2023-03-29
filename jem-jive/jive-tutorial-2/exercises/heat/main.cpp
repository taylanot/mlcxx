
#include <jem/util/Properties.h>
#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/Application.h>
#include <jive/app/InfoModule.h>
#include <jive/app/OutputModule.h>
#include <jive/app/UserconfModule.h>
#include <jive/fem/declare.h>
#include <jive/fem/ShapeModule.h>
#include <jive/fem/InputModule.h>
#include <jive/fem/InitModule.h>
#include <jive/model/declare.h>
#include <jive/femodel/declare.h>
#include <jive/mesh/declare.h>
#include <jive/mesh/MeshgenModule.h>
#include <jive/implict/Park3Module.h>
#include <jive/implict/LinsolveModule.h>
#include <jive/gl/declare.h>
#include <jive/gl/FemViewModule.h>

using jem::Ref;
using jem::util::Properties;
using jive::app::Module;

//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------

Ref<Module> mainModule ()
{
  using jem::newInstance;
  using jive::app::ChainModule;
  using jive::app::ControlModule;
  using jive::app::ReportModule;
  using jive::app::InfoModule;
  using jive::app::OutputModule;
  using jive::app::UserconfModule;
  using jive::fem::ShapeModule;
  using jive::fem::InputModule;
  using jive::fem::InitModule;
  using jive::mesh::MeshgenModule;
  using jive::implict::Park3Module;
  using jive::implict::LinsolveModule;
  using jive::gl::FemViewModule;

  Ref<ChainModule>  chain = newInstance<ChainModule> ();


  jive::mesh    :: declareGeometries ();
  jive::fem     :: declareMBuilders  ();
  jive::model   :: declareModels     ();
  jive::femodel :: declareModels     ();
  jive::gl      :: declareModules    ();

  chain->pushBack ( newInstance<MeshgenModule>   () );
  chain->pushBack ( newInstance<InputModule>     () );
  chain->pushBack ( newInstance<ShapeModule>     ( "" ) );
  chain->pushBack ( newInstance<InitModule>      () );
  //chain->pushBack ( newInstance<InfoModule>      () );
  chain->pushBack ( newInstance<ControlModule>   () );

  chain->pushBack (
    newInstance<Park3Module> (
      "park3",
      newInstance<LinsolveModule> ()
    )
  );

  //chain->pushBack ( newInstance<FemViewModule>   () );
  chain->pushBack ( newInstance<OutputModule>    () );
  chain->pushBack ( newInstance<UserconfModule>  ( "extra" ) );

  return newInstance<ReportModule> ( "report", chain );
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ( int argc, char** argv )
{
  using jive::app::Application;

  return Application::pexec ( argc, argv, & mainModule );
}
