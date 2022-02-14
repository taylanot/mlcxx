/*
 *  TestModel Header 
 *
 *  Ozgur Taylan Turan, February 2022
 *
 */

#ifndef TEST_MODEL_H 
#define TEST_MODEL_H

#include <jive/model/Model.h>

using jem::Ref;
using jem::String;
using jem::util::Properties;
using jive::model::Model;

//------------------------------------------------------------------------------
//   class TestModel
//------------------------------------------------------------------------------

class TestModel : public Model
{
  public:

  TestModel
    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

  static const char* TYPE_NAME;

  virtual bool takeAction
    ( const String&     action,
      const Properties& params,
      const Properties& globdat );

  static Ref<Model> makeNew
    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

  private:

  double parameter_;

};

//------------------------------------------------------------------------------
//    declareTestModel
//------------------------------------------------------------------------------

void declareTestModel ();

#endif
