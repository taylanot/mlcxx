#ifndef IMPINGE_MODULE_H
#define IMPINGE_MODULE_H

#include <jive/app/Module.h>
#include "import.h"

//-----------------------------------------------------------------------
//   class ImpingeModule
//-----------------------------------------------------------------------

class ImpingeModule : public jive::app::Module
{
 public:

  explicit                ImpingeModule

    ( const String&         name = "impinge" );

  virtual Status          init

    ( const Properties&     conf,
      const Properties&     props,
      const Properties&     globdat );


 protected:

  virtual                ~ImpingeModule  ();

};

#endif
