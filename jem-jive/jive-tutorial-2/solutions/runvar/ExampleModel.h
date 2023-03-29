#ifndef EXAMPLE_MODEL_H
#define EXAMPLE_MODEL_H

#include <jive/model/Model.h>

using jem::Ref;
using jem::String;
using jem::util::Properties;
using jive::model::Model;

//-----------------------------------------------------------------------
//   class ExampleModel
//-----------------------------------------------------------------------

class ExampleModel : public Model
{
 public:

                          ExampleModel

    ( const String&         name );

  virtual bool            takeAction

    ( const String&         action,
      const Properties&     params,
      const Properties&     globdat );


 private:

  int                     iiter;

};

//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------

void                      declareExampleModel  ();

#endif

