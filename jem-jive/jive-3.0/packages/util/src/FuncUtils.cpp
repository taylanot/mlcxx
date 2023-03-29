
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <cctype>
#include <jem/base/rtti.h>
#include <jem/base/limits.h>
#include <jem/base/Class.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/func/FuncException.h>
#include <jem/numeric/func/UserFunc.h>
#include <jem/numeric/func/ConstantFunc.h>
#include <jem/util/None.h>
#include <jem/util/HashDictionary.h>
#include <jive/Array.h>
#include <jive/util/typedefs.h>
#include <jive/util/Globdat.h>
#include <jive/util/FuncUtils.h>
#include "private/ConstantCond.h"


JIVE_BEGIN_PACKAGE( util )


using jem::toValue;
using jem::toObject;
using jem::isInstance;
using jem::newInstance;
using jem::staticCast;
using jem::dynamicCast;
using jem::Exception;
using jem::IllegalInputException;
using jem::util::isNone;
using jem::util::None;
using jem::numeric::UserFunc;
using jem::numeric::ConstantFunc;
using jem::numeric::FuncException;


//=======================================================================
//   private helper functions
//=======================================================================

//-----------------------------------------------------------------------
//   toUserFunc
//-----------------------------------------------------------------------


inline UserFunc*  toUserFunc

  ( Function&       func )

{
  using jem::Class;

  Class*  funcType = func.getClass     ();
  Class*  userType = UserFunc::getType ();

  if ( funcType == userType || userType->isBaseOf( funcType ) )
  {
    return static_cast<UserFunc*> ( &func );
  }
  else
  {
    return 0;
  }
}


inline UserFunc*  toUserFunc

  ( Function*       func )

{
  if ( func )
  {
    return toUserFunc ( *func );
  }
  else
  {
    return 0;
  }
}


//=======================================================================
//   class FuncUtils::NameSpace
//=======================================================================


class FuncUtils::NameSpace : public UserFunc::NameSpace
{
 public:

  explicit inline         NameSpace

    ( const Properties&     globdat );

  virtual bool            getVariable

    ( double&               value,
      const String&         name )     const override;

  virtual Ref<Function>   getFunction

    ( const String&         name )     const override;

  bool                    getArrayItem

    ( double&               value,
      const String&         name )     const;

  void                    indexError

    ( const String&         name,
      idx_t                 index,
      idx_t                 ubound )   const;


 private:

  Properties              vars_;
  Properties              funcs_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline FuncUtils::NameSpace::NameSpace

  ( const Properties&  globdat ) :

    vars_  ( globdat.findProps( Globdat::VARIABLES ) ),
    funcs_ ( globdat.findProps( Globdat::FUNCTIONS ) )

{}


//-----------------------------------------------------------------------
//   getVariable
//-----------------------------------------------------------------------


bool FuncUtils::NameSpace::getVariable

  ( double&        value,
    const String&  name ) const

{
  if ( name.endsWith ( ']' ) )
  {
    return getArrayItem ( value, name );
  }

  Ref<Object>  var;

  vars_.find ( var, name );

  if ( ! var )
  {
    return false;
  }
  else
  {
    value = toFloat ( var, name );

    return true;
  }
}


//-----------------------------------------------------------------------
//   getFunction
//-----------------------------------------------------------------------


Ref<Function> FuncUtils::NameSpace::getFunction

  ( const String& name ) const

{
  Ref<Function>  func;

  funcs_.find ( func, name );

  return func;
}


//-----------------------------------------------------------------------
//   getArrayItem
//-----------------------------------------------------------------------


bool FuncUtils::NameSpace::getArrayItem

  ( double&        value,
    const String&  name ) const

{
  Ref<Object>  var;

  idx_t        mult;
  idx_t        index;
  idx_t        i;


  i = name.size() - 2;

  while ( i > 1 && std::isspace(name[i]) )
  {
    i--;
  }

  mult  = 1;
  index = 0;

  while ( i > 1 && std::isdigit(name[i]) )
  {
    if ( index > 1000_idx )
    {
      throw IllegalInputException (
        name,
        "array index too large"
      );
    }

    index += mult * (idx_t) (name[i--] - '0');
    mult  *= 10;
  }

  while ( i > 1 && std::isspace(name[i]) )
  {
    i--;
  }

  if ( name[i] != '[' )
  {
    throw IllegalInputException (
      name,
      "illegal variable name: " + name
    );
  }

  String  vname = name[slice(BEGIN,i)];

  vars_.find ( var, vname );

  if ( ! var )
  {
    return false;
  }

  if ( isInstance<Vector>( var ) )
  {
    const Vector&  v = toValue<const Vector&> ( var );

    if ( index >= v.size() )
    {
      indexError ( vname, index, v.size() );
    }

    value = v[index];

    return true;
  }

  if ( isInstance<IntVector>( var ) )
  {
    const IntVector&  v = toValue<const IntVector&> ( var );

    if ( index >= v.size() )
    {
      indexError ( vname, index, v.size() );
    }

    value = (double) v[index];

    return true;
  }

  if ( isInstance<BoolVector>( var ) )
  {
    const BoolVector&  v = toValue<const BoolVector&> ( var );

    if ( index >= v.size() )
    {
      indexError ( vname, index, v.size() );
    }

    value = v[index] ? 1.0 : 0.0;

    return true;
  }

  throw IllegalInputException (
    name,
    String::format (
      "variable `%s\' is not a floating point array, not an "
      "integer array, and not a boolean array",
      vname
    )
  );

  return false;
}


//-----------------------------------------------------------------------
//   indexError
//-----------------------------------------------------------------------


void FuncUtils::NameSpace::indexError

  ( const String&  name,
    idx_t          index,
    idx_t          ubound ) const

{
  throw IllegalInputException (
    name,
    String::format (
      "index in array `%s\' is too large (%d); valid range is [0:%d)",
      name,
      index, ubound
    )
  );
}


//=======================================================================
//   class FuncUtils::DeclareNS
//=======================================================================


class FuncUtils::DeclareNS : public UserFunc::NameSpace
{
 public:

  explicit inline         DeclareNS

    ( const Properties&     globdat );

  virtual bool            getVariable

    ( double&               value,
      const String&         name )     const override;

  virtual Ref<Function>   getFunction

    ( const String&         name )     const override;


 private:

  Properties              vars_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline FuncUtils::DeclareNS::DeclareNS

  ( const Properties&  globdat ) :

    vars_ ( globdat.findProps( Globdat::VARIABLES ) )

{}


//-----------------------------------------------------------------------
//   getVariable
//-----------------------------------------------------------------------


bool FuncUtils::DeclareNS::getVariable

  ( double&        value,
    const String&  name ) const

{
  if ( name.endsWith( ']' ) )
  {
    idx_t  i = name.rfind ( '[' );

    if ( i >= 0 )
    {
      String  vname = name[slice(BEGIN,i)].stripWhite ();

      if ( ! vars_.contains( vname ) )
      {
        vars_.set ( vname, nullptr );
      }
    }
  }
  else if ( ! vars_.contains( name ) )
  {
    // The variables "time" and "timeStep" should not be touched;
    // they will be set automatically.

    if (  name != "time" && name != "timeStep" )
    {
      vars_.set ( name, nullptr );
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   getFunction
//-----------------------------------------------------------------------


Ref<Function> FuncUtils::DeclareNS::getFunction

  ( const String& name ) const

{
  return nullptr;
}


//=======================================================================
//   class FuncUtils
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  FuncUtils::STD_ARGS = "i, t";


//-----------------------------------------------------------------------
//   newFunc
//-----------------------------------------------------------------------


Ref<Function> FuncUtils::newFunc ( double value )
{
  return newInstance<ConstantFunc> ( value );
}


Ref<Function> FuncUtils::newFunc ( const String& expr )
{
  return newInstance<UserFunc> ( STD_ARGS, expr );
}


Ref<Function> FuncUtils::newFunc

  ( const String&  args,
    const String&  expr )

{
  return newInstance<UserFunc> ( args, expr );
}


Ref<Function> FuncUtils::newFunc

  ( const String&      expr,
    const Properties&  globdat )

{
  DeclareNS      dns  ( globdat );
  Ref<UserFunc>  func = newInstance<UserFunc> ( STD_ARGS, expr );

  func->resolve   ( dns );
  func->unresolve ();

  return func;
}


Ref<Function> FuncUtils::newFunc

  ( const String&      args,
    const String&      expr,
    const Properties&  globdat )

{
  DeclareNS      dns  ( globdat );
  Ref<UserFunc>  func = newInstance<UserFunc> ( args, expr );

  func->resolve   ( dns );
  func->unresolve ();

  return func;
}


Ref<Function> FuncUtils::newFunc

  ( const String&      args,
    const String&      name,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Function>  func;

  if ( ! configFunc( func, args, name, props, globdat ) )
  {
    props.propertyError ( name, "undefined function: " + name );
  }

  if ( ! func )
  {
    props.propertyError ( name, "empty function specification" );
  }

  return func;
}


//-----------------------------------------------------------------------
//   configFunc
//-----------------------------------------------------------------------


bool FuncUtils::configFunc

  ( Ref<Function>&     func,
    const String&      args,
    const String&      name,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Object>  obj;

  if ( ! props.find( obj, name ) )
  {
    return false;
  }

  if ( isInstance<String>( obj ) )
  {
    try
    {
      func = newFunc ( args, toValue<String>( obj ), globdat );
    }
    catch ( Exception& ex )
    {
      ex.setContext ( props.getContext( name ) );
      throw;
    }

    return true;
  }

  double  x;

  if ( toValue( x, obj ) )
  {
    func = newInstance<ConstantFunc> ( x );

    return true;
  }

  bool  b;

  if ( toValue( b, obj ) )
  {
    func = newInstance<ConstantCond> ( b );

    return true;
  }

  if ( isNone( obj ) )
  {
    func = nullptr;

    return true;
  }

  props.propertyError (
    name,
    "invalid function specification; expected a string "
    "(expression), a boolean, a number, or none"
  );

  return false;
}


//-----------------------------------------------------------------------
//   configFuncs
//-----------------------------------------------------------------------


bool FuncUtils::configFuncs

  ( FuncArray&         funcs,
    const String&      args,
    const String&      name,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Object>  obj;

  if ( ! props.find( obj, name ) )
  {
    return false;
  }

  if ( isInstance<String>( obj ) )
  {
    funcs.resize ( 1 );

    try
    {
      funcs[0] = newFunc ( args, toValue<String>( obj ), globdat );
    }
    catch ( Exception& ex )
    {
      ex.setContext ( props.getContext( name ) );
      throw;
    }

    return true;
  }

  double  x;

  if ( toValue( x, obj ) )
  {
    funcs.resize ( 1 );

    funcs[0] = newInstance<ConstantFunc> ( x );

    return true;
  }

  bool  b;

  if ( toValue( b, obj ) )
  {
    funcs.resize ( 1 );

    funcs[0] = newInstance<ConstantCond> ( b );

    return true;
  }

  if ( isNone( obj ) )
  {
    funcs.resize ( 0 );

    return true;
  }

  if ( isInstance<StringVector>( obj ) )
  {
    const StringVector&  v = toValue<const StringVector&> ( obj );

    funcs.resize ( v.size() );

    try
    {
      for ( idx_t i = 0; i < v.size(); i++ )
      {
        funcs[i] = newFunc ( args, v[i], globdat );
      }
    }
    catch ( Exception& ex )
    {
      ex.setContext ( props.getContext( name ) );
      throw;
    }

    return true;
  }

  if ( isInstance<Vector>( obj ) )
  {
    const Vector&  v = toValue<const Vector&> ( obj );

    funcs.resize ( v.size() );

    for ( idx_t i = 0; i < v.size(); i++ )
    {
      funcs[i] = newInstance<ConstantFunc> ( v[i] );
    }

    return true;
  }

  if ( isInstance<IntVector>( obj ) )
  {
    const IntVector&  v = toValue<const IntVector&> ( obj );

    for ( idx_t i = 0; i < v.size(); i++ )
    {
      funcs[i] = newInstance<ConstantFunc> ( (double) v[i] );
    }

    return true;
  }

  if ( isInstance<BoolVector>( obj ) )
  {
    const BoolVector&  v = toValue<const BoolVector&> ( obj );

    for ( idx_t i = 0; i < v.size(); i++ )
    {
      funcs[i] = newInstance<ConstantCond> ( v[i] );
    }

    return true;
  }

  props.propertyError (
    name,
    "invalid function specification; expected an array of strings "
    "(expressions), booleans, numbers, or none"
  );

  return false;
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void FuncUtils::getConfig

  ( const Properties&     conf,
    const Ref<Function>&  func,
    const String&         name )

{
  ConstantFunc*  cf = dynamicCast<ConstantFunc*> ( func );

  if ( cf )
  {
    conf.set ( name, cf->value );

    return;
  }

  ConstantCond*  cc = dynamicCast<ConstantCond*> ( func );

  if ( cc )
  {
    conf.set ( name, cc->value );

    return;
  }

  if ( ! func )
  {
    conf.set ( name, None::getInstance() );
  }
  else
  {
    conf.set ( name, func->toString() );
  }
}


void FuncUtils::getConfig

  ( const Properties&  conf,
    const FuncArray&   funcs,
    const String&      name )

{
  const idx_t  n = funcs.size ();

  if ( n == 0 )
  {
    conf.set ( name, None::getInstance() );

    return;
  }

  idx_t  cfCount = 0;
  idx_t  ccCount = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    if      ( dynamicCast<ConstantFunc*>( funcs[i] ) )
    {
      cfCount++;
    }
    else if ( dynamicCast<ConstantCond*>( funcs[i] ) )
    {
      ccCount++;
    }
    else
    {
      break;
    }
  }

  if      ( cfCount == n )
  {
    Vector  v ( n );

    for ( idx_t i = 0; i < n; i++ )
    {
      v[i] = staticCast<ConstantFunc*> ( funcs[i] ) -> value;
    }

    conf.set ( name, v );
  }
  else if ( ccCount == n )
  {
    BoolVector  v ( n );

    for ( idx_t i = 0; i < n; i++ )
    {
      v[i] = staticCast<ConstantCond*> ( funcs[i] ) -> value;
    }

    conf.set ( name, v );
  }
  else
  {
    StringVector  v ( n );

    for ( idx_t i = 0; i < n; i++ )
    {
      if ( funcs[i] )
      {
        v[i] = funcs[i]->toString ();
      }
    }

    conf.set ( name, v );
  }
}


//-----------------------------------------------------------------------
//   evalExpr
//-----------------------------------------------------------------------


double FuncUtils::evalExpr

  ( const String&      expr,
    const Properties&  globdat )

{
  Ref<UserFunc>  fn = newInstance<UserFunc> ( "", expr );

  NameSpace      ns ( globdat );

  fn->resolve ( ns );

  return fn->eval ();
}


//-----------------------------------------------------------------------
//   evalFunc
//-----------------------------------------------------------------------


double FuncUtils::evalFunc

  ( Function&          func,
    const Properties&  globdat )

{
  JEM_PRECHECK2 ( func.argCount() <= 2,
                  "Function should have at least two arguments" );

  UserFunc*    ufunc   = toUserFunc ( func );
  const idx_t  argc    = func.argCount ();

  double       args[2] = { 0.0, 0.0 };


  if ( argc > 0 )
  {
    idx_t  i = 0;

    globdat.find ( i, Globdat::TIME_STEP );

    args[0] = (double) i;
  }

  if ( argc > 1 )
  {
    double  t = 0.0;

    globdat.find ( t, Globdat::TIME );

    args[1] = t;
  }

  if ( ufunc )
  {
    NameSpace  ns ( globdat );

    ufunc->unresolve ();
    ufunc->resolve   ( ns );
  }

  return func.getValue ( args );
}


double FuncUtils::evalFunc

  ( Function&          func,
    const double*      args,
    const Properties&  globdat )

{
  UserFunc*  ufunc = toUserFunc ( func );

  if ( ufunc )
  {
    NameSpace  ns ( globdat );

    ufunc->unresolve ();
    ufunc->resolve   ( ns );
  }

  return func.getValue ( args );
}


//-----------------------------------------------------------------------
//   newCond
//-----------------------------------------------------------------------


Ref<Function> FuncUtils::newCond ( bool value )
{
  return newInstance<ConstantCond> ( value );
}


Ref<Function> FuncUtils::newCond ( const char* expr )
{
  return newInstance<UserFunc> ( STD_ARGS, expr );
}


Ref<Function> FuncUtils::newCond ( const String& expr )
{
  return newInstance<UserFunc> ( STD_ARGS, expr );
}


Ref<Function> FuncUtils::newCond

  ( const String&      expr,
    const Properties&  globdat )

{
  DeclareNS      dns  ( globdat );
  Ref<UserFunc>  func = newInstance<UserFunc> ( STD_ARGS, expr );

  func->resolve   ( dns );
  func->unresolve ();

  return func;
}


//-----------------------------------------------------------------------
//   configCond
//-----------------------------------------------------------------------


bool FuncUtils::configCond

  ( Ref<Function>&     func,
    const String&      name,
    const Properties&  props,
    const Properties&  globdat )

{
  if ( configFunc( func, name, props, globdat ) )
  {
    if ( ! func )
    {
      props.propertyError ( name, "empty condition specified" );
    }

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   evalCond
//-----------------------------------------------------------------------


bool FuncUtils::evalCond

  ( Function&          cond,
    const Properties&  globdat )

{
  JEM_PRECHECK2 ( cond.argCount() <= 2,
                  "Function should have at least two arguments" );

  const idx_t  argc    = cond.argCount ();
  UserFunc*    ufunc   = toUserFunc    ( cond );

  double       args[2] = { 0.0, 0.0 };


  if ( argc > 0 )
  {
    idx_t  i = 0;

    globdat.find ( i, Globdat::TIME_STEP );

    args[0] = (double) i;
  }

  if ( argc > 1 )
  {
    double  t = 0.0;

    globdat.find ( t, Globdat::TIME );

    args[1] = t;
  }

  if ( ufunc )
  {
    NameSpace  ns ( globdat );

    ufunc->unresolve ();
    ufunc->resolve   ( ns );
  }

  double  result = cond.getValue ( args );

  return (result * result > 0.25);
}


//-----------------------------------------------------------------------
//   resolve
//-----------------------------------------------------------------------


void FuncUtils::resolve

  ( Function&          func,
    const Properties&  globdat )

{
  UserFunc*  ufunc = toUserFunc ( func );

  if ( ufunc )
  {
    NameSpace  ns ( globdat );

    ufunc->unresolve ();
    ufunc->resolve   ( ns );
  }
}


void FuncUtils::resolve

  ( const FuncArray&   funcs,
    const Properties&  globdat )

{
  const idx_t  n = funcs.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( funcs[i] )
    {
      resolve ( *funcs[i], globdat );
    }
  }
}


//-----------------------------------------------------------------------
//   declareVars
//-----------------------------------------------------------------------


void FuncUtils::declareVars

  ( Function&          func,
    const Properties&  globdat )

{
  UserFunc*  ufunc = toUserFunc ( func );

  if ( ufunc )
  {
    DeclareNS  dns ( globdat );

    try
    {
      ufunc->unresolve ();
      ufunc->resolve   ( dns );
      ufunc->unresolve ();
    }
    catch ( const FuncException& ) {}
  }
}


//-----------------------------------------------------------------------
//   getFunc
//-----------------------------------------------------------------------


Ref<Function> FuncUtils::getFunc

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  Ref<Function>  func;
  Ref<Dict>      funcs;


  if ( globdat.find( funcs, Globdat::FUNCTIONS ) )
  {
    func = dynamicCast<Function> ( funcs->get( name ) );
  }

  if ( ! func )
  {
    throw IllegalInputException (
      context,
      String::format ( "undefined function: `%s\'", name )
    );
  }

  return func;
}


//-----------------------------------------------------------------------
//   findFunc
//-----------------------------------------------------------------------


Ref<Function> FuncUtils::findFunc

  ( const String&      name,
    const Properties&  globdat )

{
  Ref<Function>  func;
  Ref<Dict>      funcs;


  if ( globdat.find( funcs, Globdat::FUNCTIONS ) )
  {
    return dynamicCast<Function> ( funcs->get( name ) );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   storeFunc
//-----------------------------------------------------------------------


void FuncUtils::storeFunc

  ( const String&         name,
    const Ref<Function>&  func,
    const Properties&     globdat )

{
  getFuncs(globdat)->insert ( name, func );
}


//-----------------------------------------------------------------------
//   getFuncs
//-----------------------------------------------------------------------


Ref<Dictionary> FuncUtils::getFuncs ( const Properties& globdat )
{
  using jem::util::HashDict;

  Ref<Dict>  funcs;

  if ( ! globdat.find( funcs, Globdat::FUNCTIONS ) )
  {
    funcs = newInstance<HashDict> ();

    globdat.set ( Globdat::FUNCTIONS, funcs );
  }

  return funcs;
}


//-----------------------------------------------------------------------
//   findFuncs
//-----------------------------------------------------------------------


Ref<Dictionary> FuncUtils::findFuncs ( const Properties& globdat )
{
  Ref<Dict>  funcs;

  globdat.find ( funcs, Globdat::FUNCTIONS );

  return funcs;
}


JIVE_END_PACKAGE( util )
