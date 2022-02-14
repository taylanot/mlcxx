/*
 *  TestModule Header 
 *
 *  Ozgur Taylan Turan, February 2022
 *
 */

#ifndef TEST_MODULE_H
#define TEST_MODULE_H

#include <jem/util/Properties.h>
#include <jive/app/Module.h>


using namespace jem;

using jem::util::Properties;
using jive::app::Module;


//------------------------------------------------------------------------------
//    class TestModule
//------------------------------------------------------------------------------


class TestModule: public Module
{

 public:

  typedef TestModule Self;
  typedef Module     Super;

  static const char* TYPE_NAME;

  explicit TestModule 
    ( const String& name = "Test" );


  virtual Status init
    ( const Properties& conf,
      const Properties& props,
      const Properties& globdat );

  static Ref<Module> makeNew
    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

 protected:

  virtual ~TestModule ();

 private:

  void configure_ 
    ( const Properties& conf,
      const Properties& props,
      const Properties& globdat );

  String message_;

};

#endif
