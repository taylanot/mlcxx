
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <utility>
#include <jem/base/rtti.h>
#include <jem/base/assert.h>
#include <jem/base/Class.h>
#include <jem/base/Clonable.h>
#include <jem/base/ParseException.h>
#include <jem/base/RuntimeException.h>
#include <jem/base/array/utilities.h>
#include <jem/io/FileReader.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/None.h>
#include <jem/util/Flex.h>
#include <jem/util/ObjFlex.h>
#include <jem/util/Tokenizer.h>
#include <jem/util/HashDictionary.h>
#include <jem/util/ObjectConverter.h>
#include <jem/util/PropertyException.h>
#include <jem/util/Properties.h>
#include "props/utilities.h"
#include "props/Errors.h"
#include "props/Traits.h"
#include "props/Parser.h"


JEM_BEGIN_PACKAGE( util )


using jem::io::Writer;
using jem::io::PrintWriter;
using jem::io::ObjectInput;
using jem::io::ObjectOutput;
using jem::util::props::joinNames;

typedef Ref<Object>  ObjRef;


//=======================================================================
//   class Properties::Init_
//=======================================================================


class Properties::Init_
{
 public:

                    Init_ ();

  static Init_      instance;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Properties::Init_ Properties::Init_::instance;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


Properties::Init_::Init_ ()
{
  // This is to make sure that the (de)serialization of arrays
  // works properly.

  Class::registerClass ( typeOf< Array<bool>   >() );
  Class::registerClass ( typeOf< Array<int>    >() );
  Class::registerClass ( typeOf< Array<idx_t>  >() );
  Class::registerClass ( typeOf< Array<double> >() );
  Class::registerClass ( typeOf< Array<String> >() );
}


//=======================================================================
//   class Properties
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Properties::MAX_DEPTH = 256;


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


Properties::Properties ( const String& name ) :

  myName_ ( name )

{}


Properties::Properties

  ( const Ref<Dict>&  dict,
    const String&     name ) :

    dict_   ( dict ),
    myName_ ( name )

{}


Properties::Properties

  ( const Ref<Dict>&   dict,
    Ref<ObjConverter>  conv,
    const String&      name ) :

    dict_   ( dict ),
    conv_   ( conv ),
    myName_ ( name )

{}


Properties::Properties ( const Properties& rhs ) :

  dict_   ( rhs.dict_   ),
  conv_   ( rhs.conv_   ),
  myName_ ( rhs.myName_ )

{
  if ( ! dict_ )
  {
    rhs.init_ ();

    dict_ = rhs.dict_;
  }
}


Properties::Properties ( Properties&& rhs ) noexcept :

  dict_   ( std::move( rhs.dict_ ) ),
  conv_   ( std::move( rhs.conv_ ) ),
  myName_ ( std::move( rhs.myName_ ) )

{}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


Properties& Properties::operator = ( const Properties& rhs )
{
  if ( this != &rhs )
  {
    if ( ! rhs.dict_ )
    {
      rhs.init_ ();
    }

    dict_   = rhs.dict_;
    conv_   = rhs.conv_;
    myName_ = rhs.myName_;
  }

  return *this;
}


Properties& Properties::operator = ( Properties&& rhs ) noexcept
{
  dict_  .swap ( rhs.dict_ );
  conv_  .swap ( rhs.conv_ );
  myName_.swap ( rhs.myName_ );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void Properties::swap ( Properties& rhs )
{
  if ( ! dict_ )
  {
    init_ ();
  }

  if ( ! rhs.dict_ )
  {
    rhs.init_ ();
  }

  dict_  .swap ( rhs.dict_   );
  conv_  .swap ( rhs.conv_   );
  myName_.swap ( rhs.myName_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Properties Properties::clone ( CloneOptions options ) const
{
  if ( ! dict_ )
  {
    return Properties ( nullptr, conv_, myName_ );
  }

  Class*         dictClass = Dict::getType    ();

  Ref<Dict>      lhsDict   = newDictionary    ();
  Ref<DictEnum>  rhsEnum   = dict_->enumerate ();

  ObjFlex        lhsStack;
  ObjFlex        rhsStack;


  lhsStack.reserve ( 8 );
  rhsStack.reserve ( 8 );

  lhsStack.pushBack ( lhsDict );
  rhsStack.pushBack ( rhsEnum );

  while ( lhsStack.size() > 0 )
  {
    lhsDict = static_cast<Dict*>     ( lhsStack.back() );
    rhsEnum = static_cast<DictEnum*> ( rhsStack.back() );

    lhsStack.popBack ();
    rhsStack.popBack ();

    while ( ! rhsEnum->atEnd() )
    {
      String  name   = rhsEnum->getKey   ();
      ObjRef  rhsObj = rhsEnum->getValue ();

      rhsEnum->toNext ();

      if ( rhsObj.getType() != ObjRef::OBJECT_TYPE )
      {
        lhsDict->insert ( name, rhsObj );
        continue;
      }

      Class*  rhsClass = rhsObj->getClass ();
      ObjRef  lhsObj;

      if ( dictClass->isBaseOf( rhsClass ) )
      {
        lhsStack.pushBack ( lhsDict );
        rhsStack.pushBack ( rhsEnum );

        if ( rhsStack.size() > MAX_DEPTH )
        {
          depthError ();
        }

        lhsObj = rhsClass->newInstance ();

        lhsDict->insert ( name, lhsObj );

        lhsDict = checkedCast<Dict> ( lhsObj );
        rhsEnum = staticCast <Dict> ( rhsObj ) -> enumerate ();

        continue;
      }

      if ( options & CLONE_STRICT )
      {
        lhsObj = rhsObj.clone ();
      }
      else
      {
        Clonable*  c = dynamic_cast<Clonable*> ( rhsObj.get() );

        if ( c )
        {
          lhsObj = c->clone ();
        }
        else
        {
          lhsObj = rhsObj;
        }
      }

      lhsDict->insert ( name, lhsObj );
    }
  }

  return Properties ( lhsDict, conv_, myName_ );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t Properties::size () const
{
  if ( ! dict_ )
  {
    return 0;
  }
  else
  {
    return dict_->size ();
  }
}


//-----------------------------------------------------------------------
//   isShared
//-----------------------------------------------------------------------


bool Properties::isShared () const
{
  if ( ! dict_ )
  {
    return false;
  }
  else
  {
    return dict_->isShared ();
  }
}


//-----------------------------------------------------------------------
//   getContents
//-----------------------------------------------------------------------


Dict* Properties::getContents () const
{
  if ( ! dict_ )
  {
    init_ ();
  }

  return dict_.get ();
}


//-----------------------------------------------------------------------
//   setContents
//-----------------------------------------------------------------------


void Properties::setContents ( const Ref<Dict>& dict )
{
  JEM_PRECHECK2 ( dict, "NULL Dictionary" );

  dict_ = dict;
}


//-----------------------------------------------------------------------
//   enumerate
//-----------------------------------------------------------------------


Ref<DictEnum> Properties::enumerate () const
{
  if ( ! dict_ )
  {
    init_ ();
  }

  return dict_->enumerate ();
}


//-----------------------------------------------------------------------
//   parseFile
//-----------------------------------------------------------------------


void Properties::parseFile

  ( const String&  fname,
    ParseOptions   options ) const

{
  using props::Errors;
  using jem::io::FileReader;

  Ref<Tokenizer>  input =

    newInstance<Tokenizer> ( newInstance<FileReader> ( fname ) );

  parseFrom ( *input, fname, options );

  if ( input->nextToken() != Tokenizer::EOF_TOKEN )
  {
    try
    {
      input->parseError ( "end of file expected" );
    }
    catch ( Exception& ex )
    {
      Errors::setContext ( ex, input->getLineNumber(), fname );

      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   parseFrom
//-----------------------------------------------------------------------


void Properties::parseFrom

  ( Tokenizer&     input,
    const String&  source,
    ParseOptions   options ) const

{
  using props::Parser;
  using props::Errors;

  Parser  parser;

  try
  {
    parser.parse ( *this, input, source, options );
  }
  catch ( Exception& ex )
  {
    Errors::setContext ( ex,
                         parser.getLineNumber (),
                         parser.getSource     () );

    throw;
  }
}


//-----------------------------------------------------------------------
//   mergeWith
//-----------------------------------------------------------------------


void Properties::mergeWith

  ( const Properties&  rhs,
    MergeOptions       options ) const

{
  if ( rhs.dict_ != this->dict_ && rhs.dict_ != nullptr )
  {
    Ref<Dict>      lhsDict = getContents          ();
    Ref<DictEnum>  rhsEnum = rhs.dict_->enumerate ();

    if ( ! mergeWith_( *lhsDict, *rhsEnum, options ) )
    {
      rhs.depthError ();
    }
  }
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void Properties::printTo ( PrintWriter& out ) const
{
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::outdent;

  if ( ! dict_ )
  {
    return;
  }

  Ref<DictEnum>  denum = dict_->enumerate ();
  idx_t          count = 0;

  ObjFlex        stack;

  while ( true )
  {
    while ( ! denum->atEnd() )
    {
      if ( count > 0 )
      {
        out.printLine ();
      }

      count++;

      String  name  = denum->getKey   ();
      ObjRef  value = denum->getValue ();

      denum->toNext ();

      if ( printValue( out, name + "\t= ", value ) )
      {
        print ( out, ';' );
        continue;
      }

      Dict*  dict = dynamicCast<Dict*> ( value );

      if ( dict )
      {
        if ( dict->size() == 0 )
        {
          print ( out, name, "\t= {};" );
        }
        else
        {
          print ( out, name, "\t=\n{", indent );

          stack.pushBack ( denum.get() );

          if ( stack.size() > MAX_DEPTH )
          {
            depthError ();
          }

          denum = dict->enumerate ();
        }

        continue;
      }

      print ( out, name, "\t= ", value.toString(), ';' );
    }

    if ( stack.size() == 0 )
    {
      break;
    }

    denum = static_cast<DictEnum*> ( stack.back() );

    stack.popBack ();

    print ( out, outdent, endl, "};" );
  }

  if ( count > 0 )
  {
    out.printLine ();
  }
}


//-----------------------------------------------------------------------
//   printValue
//-----------------------------------------------------------------------


bool Properties::printValue

  ( PrintWriter&   out,
    const String&  prefix,
    const ObjRef&  obj )

{
  using jem::util::props::printValue;

  typedef Array<bool>    BoolArray;
  typedef Array<int>     IntArray;
  typedef Array<double>  FloatArray;
  typedef Array<String>  StringArray;

  // Try to take the fast path.

  switch ( obj.getType() )
  {
  case ObjRef::NIL_TYPE:

    print ( out, prefix, "NIL" );

    return true;

  case ObjRef::BOOL_TYPE:

    print      ( out, prefix );
    printValue ( out, obj.getBool() );

    return true;

  case ObjRef::CHAR_TYPE:

    print      ( out, prefix );
    printValue ( out, obj.getChar() );

    return true;

  case ObjRef::STRING_TYPE:

    print      ( out, prefix );
    printValue ( out, obj.getString() );

    return true;

  case ObjRef::INT_TYPE:

    print      ( out, prefix );
    printValue ( out, obj.getInt() );

    return true;

  case ObjRef::FLOAT_TYPE:

    print      ( out, prefix );
    printValue ( out, obj.getFloat() );

    return true;

  case ObjRef::OBJECT_TYPE:

    break;

  default:

    return false;
  }

  // Try to convert to an Array type.

  Class*  type = obj.getClass ();

  if ( type == typeOf<BoolArray>() )
  {
    print      ( out, prefix );
    printValue ( out, toValue<const BoolArray&>( obj ) );

    return true;
  }

  if ( type == typeOf<IntArray>() )
  {
    print      ( out, prefix );
    printValue ( out, toValue<const IntArray&>( obj ) );

    return true;
  }

  if ( type == typeOf<FloatArray>() )
  {
    print      ( out, prefix );
    printValue ( out, toValue<const FloatArray&>( obj ) );

    return true;
  }

  if ( type == typeOf<StringArray>() )
  {
    print      ( out, prefix );
    printValue ( out, toValue<const StringArray&>( obj ) );

    return true;
  }

  // Try to convert to a scalar type.

  bool    b;
  char    c;
  lint    i;
  double  x;

  if ( toValue( b, obj ) )
  {
    print      ( out, prefix );
    printValue ( out, b );

    return true;
  }

  if ( toValue( c, obj ) )
  {
    print      ( out, prefix );
    printValue ( out, c );

    return true;
  }

  if ( toValue( i, obj ) )
  {
    print      ( out, prefix );
    printValue ( out, i );

    return true;
  }

  if ( toValue( x, obj ) )
  {
    print      ( out, prefix );
    printValue ( out, x );

    return true;
  }

  if ( isInstance<String>( obj ) )
  {
    print      ( out, prefix );
    printValue ( out, toValue<String>( obj ) );

    return true;
  }

  if ( isNone( obj ) )
  {
    print      ( out, prefix );
    print ( out, "none" );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   find (Object)
//-----------------------------------------------------------------------


bool Properties::find

  ( ObjRef&        value,
    const String&  name ) const

{
  if ( name.size() == 0 )
  {
    if ( ! dict_ )
    {
      init_ ();
    }

    value = dict_;

    return true;
  }
  else if ( ! dict_ )
  {
    return false;
  }
  else
  {
    const char*  s = name.addr ();
    const idx_t  n = name.size ();

    for ( idx_t i = 0; i < n; i++ )
    {
      if ( s[i] == '.' )
      {
        return findRecursive_ ( value, name, i );
      }
    }

    return dict_->find ( value, name );
  }
}


//-----------------------------------------------------------------------
//   find (primitive & array types)
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_FIND_PROPERTY
#define JEM_DEFINE_FIND_PROPERTY( Type )                \
                                                        \
bool Properties::find ( Type&          val,             \
                        const String&  name ) const     \
{                                                       \
  using props::Errors;                                  \
  using props::Traits;                                  \
                                                        \
  ObjRef  obj;                                          \
                                                        \
  if ( ! find( obj, name ) )                            \
  {                                                     \
    return false;                                       \
  }                                                     \
                                                        \
  if ( ! conv_ )                                        \
  {                                                     \
    if ( ! Traits< Type >::convert( val, obj ) )        \
    {                                                   \
      Errors::typeMismatch (                            \
        getContext (),                                  \
        joinNames  ( myName_, name ),                   \
        obj,                                            \
        typeOf< Type > ()                               \
      );                                                \
    }                                                   \
  }                                                     \
  else                                                  \
  {                                                     \
    try                                                 \
    {                                                   \
      conv_->convert ( val, obj );                      \
    }                                                   \
    catch ( const Exception& ex )                       \
    {                                                   \
      propertyError ( name, ex.what() );                \
    }                                                   \
  }                                                     \
                                                        \
  return true;                                          \
}


JEM_DEFINE_FIND_PROPERTY( bool          )
JEM_DEFINE_FIND_PROPERTY( char          )
JEM_DEFINE_FIND_PROPERTY( byte          )
JEM_DEFINE_FIND_PROPERTY( short         )
JEM_DEFINE_FIND_PROPERTY( int           )
JEM_DEFINE_FIND_PROPERTY( long          )
JEM_DEFINE_FIND_PROPERTY( lint          )
JEM_DEFINE_FIND_PROPERTY( idx_t         )
JEM_DEFINE_FIND_PROPERTY( float         )
JEM_DEFINE_FIND_PROPERTY( double        )
JEM_DEFINE_FIND_PROPERTY( String        )
JEM_DEFINE_FIND_PROPERTY( Array<bool>   )
JEM_DEFINE_FIND_PROPERTY( Array<int>    )
JEM_DEFINE_FIND_PROPERTY( Array<idx_t>  )
JEM_DEFINE_FIND_PROPERTY( Array<double> )
JEM_DEFINE_FIND_PROPERTY( Array<String> )

#undef JEM_DEFINE_FIND_PROPERTY


//-----------------------------------------------------------------------
//   find (Properties)
//-----------------------------------------------------------------------


bool Properties::find

  ( Properties&    values,
    const String&  name ) const

{
  if ( name.size() == 0 )
  {
    values = *this;

    return true;
  }

  if ( findProps_( values.dict_, name ) )
  {
    values.conv_   = conv_;
    values.myName_ = joinNames ( myName_, name );

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   find (primitive types with range checking)
//-----------------------------------------------------------------------


#define JEM_DEFINE_FIND_PROPERTY( Type, Type2 )         \
                                                        \
bool Properties::find ( Type&          value,           \
                        const String&  name,            \
                        Type           lbound,          \
                        Type           ubound ) const   \
{                                                       \
  using props::Errors;                                  \
                                                        \
  if ( ! find( value, name ) )                          \
  {                                                     \
    return false;                                       \
  }                                                     \
                                                        \
  if ( value < lbound || value > ubound )               \
  {                                                     \
    Errors::valueOutOfRange (                           \
      getContext (),                                    \
      joinNames  ( myName_, name ),                     \
      (Type2) lbound,                                   \
      (Type2) ubound                                    \
    );                                                  \
  }                                                     \
                                                        \
  return true;                                          \
}


JEM_DEFINE_FIND_PROPERTY( int,    lint   )
JEM_DEFINE_FIND_PROPERTY( long,   lint   )
JEM_DEFINE_FIND_PROPERTY( lint,   lint   )
JEM_DEFINE_FIND_PROPERTY( idx_t,  lint   )
JEM_DEFINE_FIND_PROPERTY( float,  double )
JEM_DEFINE_FIND_PROPERTY( double, double )

#undef JEM_DEFINE_FIND_PROPERTY


//-----------------------------------------------------------------------
//   find (array types with range checking)
//-----------------------------------------------------------------------


#define JEM_DEFINE_FIND_PROPERTY( Type, Type2 )         \
                                                        \
bool Properties::find ( Array<Type>&   value,           \
                        const String&  name,            \
                        Type           lbound,          \
                        Type           ubound ) const   \
{                                                       \
  using props::Errors;                                  \
                                                        \
  if ( ! find( value, name ) )                          \
  {                                                     \
    return false;                                       \
  }                                                     \
                                                        \
  if ( min( value ) < lbound ||                         \
       max( value ) > ubound )                          \
  {                                                     \
    Errors::valueOutOfRange (                           \
      getContext (),                                    \
      joinNames  ( myName_, name ),                     \
      (Type2) lbound,                                   \
      (Type2) ubound                                    \
    );                                                  \
  }                                                     \
                                                        \
  return true;                                          \
}


JEM_DEFINE_FIND_PROPERTY( int,    lint   )
JEM_DEFINE_FIND_PROPERTY( double, double )

#undef JEM_DEFINE_FIND_PROPERTY


//-----------------------------------------------------------------------
//   get (an object)
//-----------------------------------------------------------------------


ObjRef Properties::get ( const String& name ) const
{
  ObjRef  obj;

  if ( ! find( obj, name ) )
  {
    findError_ ( name );
  }

  return obj;
}


//-----------------------------------------------------------------------
//   makeProps
//-----------------------------------------------------------------------


Properties Properties::makeProps ( const String& name ) const
{
  if ( name.size() == 0 )
  {
    return *this;
  }

  if ( ! dict_ )
  {
    init_ ();
  }

  Ref<Dict>  props;
  ObjRef     obj;
  bool       found;
  idx_t      idx;

  idx = name.find ( '.' );

  if ( idx < 0 )
  {
    found = dict_->find     ( obj, name );
  }
  else
  {
    found = findRecursive_  ( obj, name, idx );
  }

  if ( found )
  {
    props = dynamicCast<Dict> ( obj );
  }

  if ( ! props )
  {
    props = newDictionary ();

    if ( idx < 0 )
    {
      dict_->insert ( name, props );
    }
    else
    {
      setRecursive_ ( String( name ), idx,
                      ObjRef( props ) );
    }
  }

  return Properties ( props, conv_, joinNames( myName_, name ) );
}


//-----------------------------------------------------------------------
//   findProps
//-----------------------------------------------------------------------


Properties Properties::findProps ( const String& name ) const
{
  if ( name.size() == 0 )
  {
    return *this;
  }

  Ref<Dict>  props;

  findProps_ ( props, name );

  return Properties ( props, conv_, joinNames( myName_, name ) );
}


//-----------------------------------------------------------------------
//   getProps
//-----------------------------------------------------------------------


Properties Properties::getProps ( const String& name ) const
{
  if ( name.size() == 0 )
  {
    return *this;
  }

  Ref<Dict>  props;

  if ( ! findProps_( props, name ) )
  {
    findError_ ( name );
  }

  return Properties ( props, conv_, joinNames( myName_, name ) );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


Array<String> Properties::listProps ( const String& prefix ) const
{
  if ( ! dict_ )
  {
    return Array<String> ();
  }

  Ref<DictEnum>  denum = dict_->enumerate ();
  String         path  = prefix;

  ObjFlex        enumStack;
  Flex<String>   nameStack;
  Flex<String>   list;


  list.reserve ( dict_->size() );

  while ( denum )
  {
    while ( ! denum->atEnd() )
    {
      String  name = denum->getKey   ();
      ObjRef  obj  = denum->getValue ();

      name = joinNames ( path, name );

      list.pushBack ( name );

      denum->toNext ();

      Dict*  dict = dynamicCast<Dict*> ( obj );

      if ( dict )
      {
        enumStack.pushBack ( denum );
        nameStack.pushBack ( path );

        if ( enumStack.size() > MAX_DEPTH )
        {
          depthError ();
        }

        path  = name;
        denum = dict->enumerate ();
      }
    }

    denum = nullptr;

    if ( enumStack.size() )
    {
      denum = static_cast<DictEnum*> ( enumStack.back() );
      path  = nameStack.back ();

      enumStack.popBack ();
      nameStack.popBack ();
    }
  }

  return Array<String> ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool Properties::contains ( const String& name ) const
{
  if      ( name.size() == 0 )
  {
    return true;
  }
  else if ( ! dict_ )
  {
    return false;
  }
  else
  {
    ObjRef  dummy;
    idx_t   idx;

    idx = name.find ( '.' );

    if ( idx < 0 )
    {
      return dict_->find    ( dummy, name );
    }
    else
    {
      return findRecursive_ ( dummy, name, idx );
    }
  }
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void Properties::erase ( const String& name ) const
{
  if ( dict_ )
  {
    idx_t  idx = name.find ( '.' );

    if ( idx < 0 )
    {
      dict_->erase    ( name );
    }
    else
    {
      eraseRecursive_ ( name, idx );
    }
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void Properties::clear () const
{
  if ( dict_ )
  {
    dict_->clear ();
  }
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Properties::getContext () const
{
  if ( myName_.size() > 0 )
  {
    return String::format ( "property set `%s\'", myName_ );
  }
  else
  {
    return String ( "unnamed property set" );
  }
}


String Properties::getContext ( const String& name ) const
{
  if ( myName_.size() > 0 )
  {
    return String::format ( "property `%s.%s\'",
                            myName_, name );
  }
  else
  {
    return String::format ( "property `%s\'", name );
  }
}


//-----------------------------------------------------------------------
//   propertyError
//-----------------------------------------------------------------------


void Properties::propertyError

  ( const String&  name,
    const String&  msg ) const

{
  throw PropertyException ( getContext( name ), msg );
}


//-----------------------------------------------------------------------
//   depthError
//-----------------------------------------------------------------------


void Properties::depthError () const
{
  throw RuntimeException (
    getContext (),
    String::format (
      "maximum depth (%d) exceeded (circular reference?)", MAX_DEPTH
    )
  );
}


//-----------------------------------------------------------------------
//   newDictionary
//-----------------------------------------------------------------------


Ref<Dict> Properties::newDictionary () const
{
  if ( ! dict_ )
  {
    return newInstance<HashDict> ();
  }
  else
  {
    Class*  type = dict_->getClass ();

    return checkedCast<Dict> ( type->newInstance() );
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


inline void Properties::init_ () const
{
  const_cast<Self*>( this )->dict_ = newInstance<HashDict> ();
}


//-----------------------------------------------------------------------
//   set_ (Object)
//-----------------------------------------------------------------------


void Properties::set_

  ( String&&  name,
    ObjRef&&  value ) const

{
  JEM_PRECHECK2 ( name.size() > 0, "empty property name" );

  const idx_t  n = name.size ();
  const char*  s = name.addr ();

  if ( ! dict_ )
  {
    init_ ();
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( s[i] == '.' )
    {
      setRecursive_ ( std::move( name ), i,
                      std::move( value ) );

      return;
    }
  }

  dict_->insert ( std::move( name ),
                  std::move( value ) );
}


//-----------------------------------------------------------------------
//   set_ (primitive & array types)
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_SET_PROPERTY
#define JEM_DEFINE_SET_PROPERTY( Type )             \
                                                    \
void Properties::set_ ( String&&  name,             \
                        Type      value ) const     \
{                                                   \
  set_ ( std::move( name ), toObject( value ) );    \
}


JEM_DEFINE_SET_PROPERTY( bool                 )
JEM_DEFINE_SET_PROPERTY( char                 )
JEM_DEFINE_SET_PROPERTY( byte                 )
JEM_DEFINE_SET_PROPERTY( short                )
JEM_DEFINE_SET_PROPERTY( int                  )
JEM_DEFINE_SET_PROPERTY( long                 )
JEM_DEFINE_SET_PROPERTY( lint                 )
JEM_DEFINE_SET_PROPERTY( idx_t                )
JEM_DEFINE_SET_PROPERTY( float                )
JEM_DEFINE_SET_PROPERTY( double               )
JEM_DEFINE_SET_PROPERTY( const String&        )
JEM_DEFINE_SET_PROPERTY( const Array<bool>&   )
JEM_DEFINE_SET_PROPERTY( const Array<int>&    )
JEM_DEFINE_SET_PROPERTY( const Array<idx_t>&  )
JEM_DEFINE_SET_PROPERTY( const Array<double>& )
JEM_DEFINE_SET_PROPERTY( const Array<String>& )

#undef JEM_DEFINE_SET_PROPERTY


//-----------------------------------------------------------------------
//   set_ (character string)
//-----------------------------------------------------------------------


void Properties::set_

  ( String&&     name,
    const char*  value ) const

{
  set_ ( std::move( name ), toObject( String( value ) ) );
}


//-----------------------------------------------------------------------
//   set_ (Properties)
//-----------------------------------------------------------------------


void Properties::set_

  ( String&&           name,
    const Properties&  values ) const

{
  JEM_PRECHECK2 ( name.size() > 0, "empty property name" );

  idx_t  idx = name.find ( '.' );

  if ( ! dict_ )
  {
    init_ ();
  }

  if ( ! values.dict_ )
  {
    values.init_ ();
  }

  if ( idx < 0 )
  {
    dict_->insert ( name, values.dict_ );
  }
  else
  {
    setRecursive_ ( std::move( name ), idx,
                    ObjRef( values.dict_ ) );
  }
}


//-----------------------------------------------------------------------
//   setRecursive_
//-----------------------------------------------------------------------


void Properties::setRecursive_

  ( String&&       name,
    idx_t          last,
    Ref<Object>&&  value ) const

{
  Dict*   props = dict_.get ();
  idx_t   first = 0;

  ObjRef  obj;


  while ( last >= 0 )
  {
    if ( last > first )
    {
      bool  found = props->find ( obj, name[slice(first,last)] );

      if ( ! found || ! dynamicCast<Dict*>( obj ) )
      {
        obj = newDictionary ();

        props->insert ( name[slice(first,last)],
                        ObjRef( obj ) );
      }

      props = staticCast<Dict*> ( obj );
    }

    first = last + 1;
    last  = name.find ( '.', first );
  }

  props->insert ( name[slice(first,END)],
                  std::move( value ) );
}


//-----------------------------------------------------------------------
//   findProps_
//-----------------------------------------------------------------------


bool Properties::findProps_

  ( Ref<Dict>&     props,
    const String&  name ) const

{
  using props::Errors;

  if ( ! dict_ )
  {
    return false;
  }

  ObjRef  obj;
  bool    found;
  idx_t   idx;

  idx = name.find ( '.' );

  if ( idx < 0 )
  {
    found = dict_->find     ( obj, name );
  }
  else
  {
    found = findRecursive_  ( obj, name, idx );
  }

  if ( ! found )
  {
    return false;
  }

  Dict*  dict = dynamicCast<Dict*> ( obj );

  if ( dict )
  {
    props = dict;

    return true;
  }

  if ( isNone( obj ) )
  {
    props = newDictionary ();

    return true;
  }

  Errors::typeMismatch (
    getContext (),
    joinNames  ( myName_, name ),
    obj,
    Dict::getType ()
  );

  return false;
}


//-----------------------------------------------------------------------
//   findRecursive_
//-----------------------------------------------------------------------


bool Properties::findRecursive_

  ( ObjRef&        value,
    const String&  name,
    idx_t          last ) const

{
  Dict*   props = dict_.get ();
  idx_t   first = 0;

  ObjRef  obj;


  while ( last >= 0 )
  {
    if ( last > first )
    {
      if ( ! props->find( obj, name[slice(first,last)] ) )
      {
        return false;
      }

      props = dynamicCast<Dict*> ( obj );

      if ( ! props )
      {
        return false;
      }
    }

    first = last + 1;
    last  = name.find ( '.', first );
  }

  return props->find ( value, name[slice(first,END)] );
}


//-----------------------------------------------------------------------
//   eraseRecursive_
//-----------------------------------------------------------------------


void Properties::eraseRecursive_

  ( const String&  name,
    idx_t          last ) const

{
  Dict*   props = dict_.get ();
  idx_t   first = 0;

  ObjRef  obj;


  while ( last >= 0 )
  {
    if ( last > first )
    {
      if ( ! props->find( obj, name[slice(first,last)] ) )
      {
        return;
      }

      props = dynamicCast<Dict*> ( obj );

      if ( ! props )
      {
        return;
      }
    }

    first = last + 1;
    last  = name.find ( '.', first );
  }

  props->erase ( name[slice(first,END)] );
}


//-----------------------------------------------------------------------
//   typeError_
//-----------------------------------------------------------------------


void Properties::typeError_

  ( const String&       name,
    const Ref<Object>&  obj,
    Class*              type ) const

{
  props::Errors::typeMismatch (
    getContext (),
    joinNames  ( myName_, name ),
    obj, type
  );
}


//-----------------------------------------------------------------------
//   findError_
//-----------------------------------------------------------------------


void Properties::findError_ ( const String&  name ) const
{
  props::Errors::noSuchProperty (
    getContext (),
    joinNames  ( myName_, name )
  );
}


//-----------------------------------------------------------------------
//   mergeWith_
//-----------------------------------------------------------------------


bool Properties::mergeWith_

  ( Dict&         lhs,
    DictEnum&     rhs,
    MergeOptions  options,
    int           depth ) const

{
  if ( (++depth) > MAX_DEPTH )
  {
    return false;
  }

  while ( ! rhs.atEnd() )
  {
    String  name   = rhs.getKey   ();
    ObjRef  rhsObj = rhs.getValue ();
    ObjRef  lhsObj;

    rhs.toNext ();

    if ( ! lhs.find( lhsObj, name ) )
    {
      if ( dynamicCast<Dict*>( rhsObj ) )
      {
        Ref<Dict>      lhsDict = newDictionary ();
        Ref<DictEnum>  rhsEnum =

          staticCast<Dict> ( rhsObj )->enumerate ();

        if ( ! mergeWith_( *lhsDict, *rhsEnum, options, depth ) )
        {
          return false;
        }

        lhs.insert ( name, lhsDict );
      }
      else
      {
        lhs.insert ( name, rhsObj );
      }

      continue;
    }

    if ( lhsObj == rhsObj )
    {
      continue;
    }

    if ( lhsObj == nullptr || rhsObj == nullptr )
    {
      if ( options & MERGE_REPLACE )
      {
        lhs.insert ( name, rhsObj );
      }

      continue;
    }

    if ( dynamicCast<Dict*>( lhsObj ) &&
         dynamicCast<Dict*>( rhsObj ) )
    {
      Ref<Dict>      lhsDict = staticCast<Dict> ( lhsObj );
      Ref<DictEnum>  rhsEnum =

        staticCast<Dict> ( rhsObj )->enumerate ();

      if ( ! mergeWith_( *lhsDict, *rhsEnum, options, depth ) )
      {
        return false;
      }
    }
    else if ( options & MERGE_REPLACE )
    {
      lhs.insert ( name, rhsObj );
    }
  }

  return true;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


void                  print

  ( Writer&             out,
    const Properties&   props )

{
  using jem::io::flush;
  using jem::io::indent;

  Ref<PrintWriter>  prn = dynamic_cast<PrintWriter*> ( &out );

  if ( ! prn )
  {
    prn = newInstance<PrintWriter> ( Ref<Writer>( &out ) );
  }

  print ( *prn, indent, props, flush );
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


void                  decode

  ( ObjectInput&        in,
    Properties&         props )

{
  Ref<Dict>  dict = checkedCast<Dict> ( in.decodeObject() );

  String     name;

  decode ( in, name );

  props = Properties ( dict, name );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


void                  encode

  ( ObjectOutput&       out,
    const Properties&   props )

{
  out.encodeObject ( Ref<Dict>( props.getContents() ) );

  encode ( out, props.getName() );
}


JEM_END_PACKAGE( util )
