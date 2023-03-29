#ifndef STEP_MODULE_H
#define STEP_MODULE_H

#include <jive/app/Module.h>

using jem::Ref;
using jem::String;
using jem::util::Properties;
using jive::app::Module;

//-----------------------------------------------------------------------
//   class StepModule
//-----------------------------------------------------------------------

class StepModule : public Module
{
 public:

                          StepModule

    ( const String&         name = "step" );

  virtual Status          run

    ( const Properties&     globdat );

};

#endif

