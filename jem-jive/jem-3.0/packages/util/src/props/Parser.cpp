
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


#include <cctype>
#include <cstdlib>
#include <jem/base/limits.h>
#include <jem/base/Time.h>
#include <jem/base/Class.h>
#include <jem/base/System.h>
#include <jem/base/CString.h>
#include <jem/io/FileName.h>
#include <jem/io/FileReader.h>
#include <jem/io/FileOpenException.h>
#include <jem/util/None.h>
#include <jem/util/Tokenizer.h>
#include <jem/util/Dictionary.h>
#include <jem/util/PropertyException.h>
#include "Traits.h"
#include "Errors.h"
#include "Parser.h"


JEM_BEGIN_PACKAGE( util )
JEM_BEGIN_NAMESPACE( props )


using jem::io::FileName;


//=======================================================================
//   class Parser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Parser::NONE_TOKEN        = Tokenizer::NULL_TOKEN - 1;
const int  Parser::INCLUDE_TOKEN     = NONE_TOKEN            - 1;
const int  Parser::SINCLUDE_TOKEN    = INCLUDE_TOKEN         - 1;

const int  Parser::ASSIGN_PROPERTY   = 1;
const int  Parser::EXTEND_PROPERTY   = 2;

const int  Parser::MAX_INCLUDE_DEPTH = 32;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Parser::Parser ()
{
  strbuf_.reserve ( 32 );
}


Parser::~Parser ()
{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void Parser::parse

  ( const Properties&  props,
    Tokenizer&         input,
    const String&      source,
    Options            options )

{
  Tokenizer::Options  savedOptions = input.getOptions ();


  input_   = & input;
  source_  = source;
  options_ = options;

  inputStack_.clear    ();
  propsStack_.clear    ();
  propsStack_.pushBack ( props );

  input.setOptions ( Tokenizer::PARSE_SYMBOLS        |
                     Tokenizer::PARSE_STRINGS        |
                     Tokenizer::PARSE_BOOLEANS       |
                     Tokenizer::PARSE_SIGNED_NUMBERS |
                     Tokenizer::PARSE_OPERATORS      |
                     Tokenizer::CATENATE_STRINGS );

  try
  {
    parse_ ();
  }
  catch ( ... )
  {
    input.setOptions ( savedOptions );
    throw;
  }

  input.setOptions   ( savedOptions );

  input_  = nullptr;
  source_ = "";
}


//-----------------------------------------------------------------------
//   parse_
//-----------------------------------------------------------------------


void Parser::parse_ ()
{
  String  name;
  int     mode;
  int     token;


  while ( true )
  {
    token = nextToken_ ();

    if ( token == Tokenizer::EOF_TOKEN )
    {
      if ( inputStack_.size() == 0 )
      {
        break;
      }

      InputFrame_&  frame = inputStack_.back ();

      input_  = frame.input;
      source_ = frame.source;

      inputStack_.popBack ();

      continue;
    }

    if ( token == Tokenizer::CLOSE_BRACE_TOKEN &&
         propsStack_.size() > 1 )
    {
      propsStack_.popBack ();

      if ( nextToken_() != Tokenizer::SEMI_COLON_TOKEN )
      {
        input_->parseError ( "expected a `;\'" );
      }

      continue;
    }

    if ( token == INCLUDE_TOKEN || token == SINCLUDE_TOKEN )
    {
      parseInclude_ ( token );

      continue;
    }

    if ( token != Tokenizer::SYMBOL_TOKEN &&
         token != Tokenizer::DOT_TOKEN )
    {
      break;
    }

    name  = parseName_ ( token );
    token = nextToken_ ();
    mode  = ASSIGN_PROPERTY;

    if      ( token == Tokenizer::PLUS_IS_TOKEN )
    {
      mode = EXTEND_PROPERTY;
    }
    else if ( token != Tokenizer::ASSIGN_TOKEN )
    {
      input_->parseError ( "expected a `=\' or `+=\'" );
    }

    parseProperty_ ( name, mode );
  }

  if ( propsStack_.size() > 1 )
  {
    input_->parseError ( "expected a `}\'" );
  }

  if ( inputStack_.size() > 0 )
  {
    input_->parseError ();
  }

  input_->pushBackToken ();
}


//-----------------------------------------------------------------------
//   parseName_
//-----------------------------------------------------------------------


String Parser::parseName_ ( int token )
{
  String  name;

  if ( token == Tokenizer::SYMBOL_TOKEN )
  {
    name  = input_->getSymbol  ();
    token =         nextToken_ ();
  }

  if ( token == Tokenizer::DOT_TOKEN )
  {
    strbuf_.clear ();

    strbuf_ += name;

    while ( token == Tokenizer::DOT_TOKEN )
    {
      if ( nextToken_() != Tokenizer::SYMBOL_TOKEN )
      {
        input_->parseError ( "expected a symbol" );
      }

      name = input_->getSymbol ();

      strbuf_ += '.';
      strbuf_ += name;

      token = nextToken_ ();
    }

    name = strbuf_.toString ();
  }

  input_->pushBackToken ();

  if ( options_ & Properties::PARSE_TO_UPPER )
  {
    return name.toUpper ();
  }
  else
  {
    return name;
  }
}


//-----------------------------------------------------------------------
//   parseProperty_
//-----------------------------------------------------------------------


void Parser::parseProperty_

  ( const String&  name,
    int            mode )

{
  int     token = nextToken_ ();

  ObjRef  obj;


  if      ( token == NONE_TOKEN )
  {
    obj = None::getInstance ();
  }
  else if ( token == Tokenizer::BOOLEAN_TOKEN )
  {
    obj = toObject ( input_->getBoolean() );
  }
  else if ( token == Tokenizer::INTEGER_TOKEN )
  {
    obj = toObject ( input_->getInteger() );
  }
  else if ( token == Tokenizer::FLOAT_TOKEN )
  {
    obj = toObject ( input_->getFloat() );
  }
  else if ( token == Tokenizer::STRING_TOKEN )
  {
    String  str = input_->getString ();

    if ( input_->getStringType() == '\"' &&
         str.find( '$' )         >= 0 )
    {
      expandString_ ( str );
    }

    obj   = toObject   ( str );
    token = nextToken_ ();

    if ( token == Tokenizer::OPEN_BRACE_TOKEN )
    {
      Ref<Dict>  dict;

      if ( mode == EXTEND_PROPERTY )
      {
        dict = findProperties_ ( name );
      }

      if ( ! dict )
      {
        dict = newProperties_ ();

        setProperty_ ( name, dict );
      }

      propsStack_.pushBack ( Properties( dict ) );
      setProperty_         ( "type", obj );

      return;
    }

    input_->pushBackToken ();
  }
  else if ( token == Tokenizer::SYMBOL_TOKEN ||
            token == Tokenizer::DOT_TOKEN )
  {
    obj = getProperty_ ( parseName_( token ) );
    obj = obj.clone    ();
  }
  else if ( token == Tokenizer::OPEN_BRACKET_TOKEN )
  {
    obj = parseArray_ ();
  }
  else if ( token == Tokenizer::OPEN_BRACE_TOKEN )
  {
    Ref<Dict>  dict;

    if ( mode == EXTEND_PROPERTY )
    {
      dict = findProperties_ ( name );
    }

    if ( ! dict  )
    {
      dict = newProperties_ ();

      setProperty_ ( name, dict );
    }

    propsStack_.pushBack ( Properties( dict ) );

    return;
  }
  else
  {
    input_->parseError ();
  }

  if ( nextToken_() != Tokenizer::SEMI_COLON_TOKEN )
  {
    input_->parseError ( "expected a `;\'" );
  }

  if ( mode == EXTEND_PROPERTY )
  {
    obj = extendProperty_ ( name, obj );
  }

  setProperty_ ( name, obj );
}


//-----------------------------------------------------------------------
//   parseArray_
//-----------------------------------------------------------------------


Ref<Object> Parser::parseArray_ ()
{
  int  token = nextToken_ ();

  if ( token == Tokenizer::CLOSE_BRACKET_TOKEN )
  {
    return None::getInstance ();
  }

  input_->pushBackToken ();

  if      ( token == Tokenizer::BOOLEAN_TOKEN )
  {
    return toObject ( Traits< Array<bool> >  ::parse( *input_ ) );
  }
  else if ( token == Tokenizer::INTEGER_TOKEN )
  {
    return toObject ( Traits< Array<int> >   ::parse( *input_ ) );
  }
  else if ( token == Tokenizer::FLOAT_TOKEN )
  {
    return toObject ( Traits< Array<double> >::parse( *input_ ) );
  }
  else if ( token == Tokenizer::STRING_TOKEN )
  {
    return parseStringArray_ ();
  }
  else
  {
    input_->parseError ();
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   parseStringArray_
//-----------------------------------------------------------------------


Ref<Object> Parser::parseStringArray_ ()
{
  Tokenizer&  in = * input_;

  ArrayBuffer<String>  buf;

  while ( true )
  {
    String  str = Traits<String>::parse ( in );

    if ( buf.size() == maxOf( buf.size() ) )
    {
      Errors::arraySizeOverflow ( getLineNumber() );
    }

    if ( in.getStringType() == '\"' && str.find( '$' ) >= 0 )
    {
      expandString_ ( str );
    }

    buf.pushBack ( std::move( str ) );

    int  token = in.nextToken ();

    if ( token == Tokenizer::CLOSE_BRACKET_TOKEN )
    {
      break;
    }

    if ( token == Tokenizer::COMMA_TOKEN )
    {
      continue;
    }

    in.parseError ( "expected a `]\' or `,\'" );
  }

  return toObject ( buf.toArray() );
}


//-----------------------------------------------------------------------
//   parseInclude_
//-----------------------------------------------------------------------


void Parser::parseInclude_ ( int token )
{
  using jem::io::FileReader;
  using jem::io::IOException;

  InputFrame_  frame;
  InputFrame_  next;
  String       dir;


  next.source = getSource ();

  if ( nextToken_() != Tokenizer::STRING_TOKEN )
  {
    input_->parseError ( "expected a string" );
  }

  next.source = input_->getString ();

  if ( input_->getStringType() == '\"' &&
       next.source.find( '$' ) >= 0 )
  {
    expandString_ ( next.source );
  }

  if ( nextToken_() != Tokenizer::SEMI_COLON_TOKEN )
  {
    input_->parseError ( "expected a `;\'" );
  }

  if ( inputStack_.size() > MAX_INCLUDE_DEPTH )
  {
    Errors::illegalInput (
      getLineNumber  (),
      source_,
      String::format ( "max include depth (%d) exceeded",
                       MAX_INCLUDE_DEPTH )
    );
  }

  if ( ! FileName::isAbsolute( next.source ) )
  {
    next.source = FileName::join ( FileName::getDirName ( source_ ),
                                   next.source );

    next.source = FileName::simplify ( next.source );
  }

  try
  {
    next.input = newInstance<Tokenizer> (
      newInstance<FileReader> ( next.source )
    );
  }
  catch ( const IOException& ex )
  {
    if ( token == SINCLUDE_TOKEN )
    {
      return;
    }

    Errors::illegalInput ( getLineNumber(), source_, ex.what() );
  }

  next.input->setOptions ( input_->getOptions() );

  frame.input  = input_;
  frame.source = source_;

  inputStack_.pushBack ( std::move( frame ) );

  input_  = next.input;
  source_ = next.source;
}


//-----------------------------------------------------------------------
//   getProperty_
//-----------------------------------------------------------------------


Ref<Object> Parser::getProperty_ ( const String& name )
{
  ObjRef  obj;

  if ( ! findProperty_( obj, name ) )
  {
    Errors::noSuchProperty ( String( getLineNumber() ), name );
  }

  return obj;
}


bool Parser::findProperty_

  ( ObjRef&        obj,
    const String&  name )

{
  const idx_t  n = name.size ();

  Ref<Dict>    dict;
  String       part;
  idx_t        pos;
  idx_t        i;


  if ( n == 0 )
  {
    return false;
  }

  pos = 0;

  if ( name[0] == '.' )
  {
    obj = propsStack_[0].getContents ();
    pos = 0;
  }
  else
  {
    i = name.find ( '.' );

    if ( i < 0 )
    {
      part = name;
      pos  = n;
    }
    else
    {
      part = name[slice(BEGIN,i)];
      pos  = i;
    }

    for ( i = propsStack_.size() - 1; i >= 0; i-- )
    {
      if ( propsStack_[i].find( obj, part ) )
      {
        break;
      }
    }

    if ( i < 0 )
    {
      return false;
    }
  }

  while ( pos < n )
  {
    dict = dynamicCast<Dict> ( obj );

    if ( ! dict )
    {
      Errors::nonAggregateType ( String( getLineNumber() ),
                                 name[slice(BEGIN,pos)] );
    }

    JEM_ASSERT ( name[pos] == '.' );

    pos++;
    i = name.find ( '.', pos );

    if ( i < 0 )
    {
      part = name[slice(pos,END)];
      pos  = n;
    }
    else
    {
      part = name[slice(pos,i)];
      pos  = i;
    }

    if ( ! dict->find( obj, part ) )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   findProperties_
//-----------------------------------------------------------------------


Ref<Dict> Parser::findProperties_ ( const String& name )
{
  Ref<Dict>  dict;
  ObjRef     obj;

  if ( findProperty_( obj, name ) )
  {
    dict = dynamicCast<Dict> ( obj );

    if ( ! dict )
    {
      Errors::typeMismatch ( String( getLineNumber() ),
                             name, obj, Dict::getType() );
    }
  }

  return dict;
}


//-----------------------------------------------------------------------
//   setProperty_
//-----------------------------------------------------------------------


void Parser::setProperty_

  ( const String&  name,
    const ObjRef&  value )

{
  if ( name[0] == '.' )
  {
    setProperty_ ( propsStack_[0], name[slice(1,END)], value );
  }
  else
  {
    setProperty_ ( propsStack_.back(), name, value );
  }
}


void Parser::setProperty_

  ( Properties&    props,
    const String&  name,
    const ObjRef&  value )

{
  if ( ! (options_ & Properties::PARSE_STRICT) )
  {
    props.set ( name, value );
  }
  else
  {
    ObjRef  obj;

    if ( ! props.find( obj, name ) )
    {
      Errors::noSuchProperty ( String( getLineNumber() ), name );
    }

    if ( obj == nullptr || isNone( obj ) )
    {
      props.set ( name, value );
      return;
    }

    Class*  target = obj.getClass ();

    if ( target != value.getClass() )
    {
      // Are they really incompatible?

      if ( ! target->isInstance( value ) )
      {
        Errors::typeMismatch ( String( getLineNumber() ),
                               name, value, target );
      }
    }

    props.set ( name, value );
  }
}


//-----------------------------------------------------------------------
//   newProperties_
//-----------------------------------------------------------------------


Ref<Dict> Parser::newProperties_ ()
{
  Ref<Dict>  dict = propsStack_.back().getContents ();

  return checkedCast<Dict> ( dict->getClass()->newInstance() );
}


//-----------------------------------------------------------------------
//   checkName_
//-----------------------------------------------------------------------


bool Parser::checkName_ ( const String& name )
{
  const idx_t  n = name.size ();

  idx_t        i = 0;

  if ( n > 0 && name[0] == '.' )
  {
    i++;
  }

  while ( true )
  {
    if ( i == n || ! (std::isalpha(name[i]) || name[i] == '_') )
    {
      return false;
    }

    for ( i++; i < n; i++ )
    {
      if ( ! (std::isalnum(name[i]) || name[i] == '_') )
      {
        break;
      }
    }

    if ( i == n )
    {
      break;
    }

    if ( name[i] != '.' )
    {
      return false;
    }

    i++;
  }

  return true;
}


//-----------------------------------------------------------------------
//   expandString_
//-----------------------------------------------------------------------


void Parser::expandString_ ( String& str )
{
  const idx_t  n = str.size ();

  ObjRef       obj;
  String       sub;
  idx_t        pos;
  idx_t        i, j, k;


  strbuf_.clear ();

  pos = 0;

  while ( pos < n )
  {
    if ( str[pos] != '$' )
    {
      strbuf_ += str[pos];
      pos++;

      continue;
    }

    i = pos + 1;

    if ( i == n )
    {
      expansionError_ ( str[slice(pos,END)] );
    }

    if ( str[i] == '$' )
    {
      strbuf_ += '$';
      pos      = i + 1;

      continue;
    }

    if ( str[i] != '(' )
    {
      expansionError_ ( str[slice(pos,END)] );
    }

    i++;

    while ( i < n && std::isspace( str[i] ) )
    {
      i++;
    }

    j = i;

    while ( j < n && str[j] != ')' )
    {
      j++;
    }

    if ( j == n )
    {
      expansionError_ ( str[slice(pos,END)] );
    }

    k = j;
    j = j + 1;

    while ( k > i && std::isspace(str[k - 1]) )
    {
      k--;
    }

    if ( i == k )
    {
      expansionError_ ( str[slice(pos,j)] );
    }

    sub = str[slice(i,k)];

    if ( sub.size()  >  1   &&
         sub.front() == '_' && sub.back() == '_' )
    {
      sub = expandBuiltin_ ( sub );
    }
    else
    {
      if ( ! checkName_( sub ) )
      {
        Errors::illegalInput (
          getLineNumber  (),
          getSource      (),
          String::format ( "invalid property name: `%s\'", sub )
        );
      }

      try
      {
        obj = getProperty_ ( sub );

        if ( ! obj )
        {
          sub = "";
        }
        else
        {
          sub = obj.toString ();
        }
      }
      catch ( const PropertyException& )
      {
        // Property not found; check whether an environment variable
        // with the specified name has been defined.

        const char*  envstr = ::getenv ( makeCString( sub ) );

        if ( ! envstr )
        {
          throw;
        }

        sub = clone ( envstr );
      }
    }

    pos      = j;
    strbuf_ += sub;
  }

  str = strbuf_.toString ();
}


//-----------------------------------------------------------------------
//   expandBuiltin_
//-----------------------------------------------------------------------


String Parser::expandBuiltin_ ( const String& str ) const
{
  if      ( str == "_PATH_" )
  {
    return getSource ();
  }
  else if ( str == "_FILE_" )
  {
    return FileName::getFileName ( getSource() );
  }
  else if ( str == "_DIR_" )
  {
    return FileName::getDirName  ( getSource() );
  }
  else if ( str == "_USER_" )
  {
    return System::getProperty ( "user.name" );
  }
  else if ( str == "_HOME_" )
  {
    return System::getProperty ( "user.home" );
  }
  else if ( str == "_HOST_" )
  {
    return System::getProperty ( "host.name" );
  }
  else if ( str == "_TIME_" )
  {
    return Time::now().toString ( "%H:%M:%S" );
  }
  else if ( str == "_DATE_" )
  {
    return Time::now().toString ( "%c" );
  }
  else if ( str == "_ERIKJAN_" )
  {
    return "God";
  }
  else if ( str == "_MARTIJN_" )
  {
    return "Lesser God";
  }
  else if ( str == "_JORIS_" )
  {
    return "Greetings master Joris!";
  }
  else
  {
    Errors::illegalInput (
      getLineNumber  (),
      getSource      (),
      String::format ( "unknown built-in string expansion: `%s\'",
                       str )
    );
  }

  return "unknown";
}


//-----------------------------------------------------------------------
//   expansionError_
//-----------------------------------------------------------------------


void Parser::expansionError_ ( const String& str ) const
{
  Errors::illegalInput (
    getLineNumber  (),
    getSource      (),
    String::format ( "invalid string expansion: `%s\'", str )
  );
}


//-----------------------------------------------------------------------
//   extendProperty_
//-----------------------------------------------------------------------


Ref<Object> Parser::extendProperty_

  ( const String&  name,
    const ObjRef&  rhs )

{
  typedef Array<bool>    BoolArray;
  typedef Array<int>     IntArray;
  typedef Array<double>  FloatArray;
  typedef Array<String>  StringArray;

  ObjRef  lhs;


  findProperty_ ( lhs, name );

  if ( lhs == nullptr || isNone( lhs ) )
  {
    return rhs;
  }

  if ( isInstance<bool>( lhs ) &&
       isInstance<bool>( rhs ) )
  {
    return toObject ( toValue<bool>( lhs ) ||
                      toValue<bool>( rhs ) );
  }

  if ( isInstance<idx_t>( lhs ) &&
       isInstance<idx_t>( rhs ) )
  {
    return toObject ( toValue<idx_t>( lhs ) +
                      toValue<idx_t>( rhs ) );
  }

  if ( isInstance<double>( lhs ) &&
       isInstance<double>( rhs ) )
  {
    return toObject ( toValue<double>( lhs ) +
                      toValue<double>( rhs ) );
  }

  if ( isInstance<String>( lhs ) &&
       isInstance<String>( rhs ) )
  {
    return toObject ( toValue<String>( lhs ) +
                      toValue<String>( rhs ) );
  }

  if ( isInstance<BoolArray>( lhs ) )
  {
    return extendArray_ ( name, toValue<const BoolArray&>  ( lhs ),
                          rhs );
  }

  if ( isInstance<IntArray>( lhs ) )
  {
    return extendArray_ ( name, toValue<const IntArray&>   ( lhs ),
                          rhs );
  }

  if ( isInstance<FloatArray>( lhs ) )
  {
    return extendArray_ ( name, toValue<const FloatArray&> ( lhs ),
                          rhs );
  }

  if ( isInstance<StringArray>( lhs ) )
  {
    return extendArray_ ( name, toValue<const StringArray&>( lhs ),
                          rhs );
  }

  if ( dynamicCast<Dict>( lhs ) &&
       dynamicCast<Dict>( rhs ) )
  {
    Properties  lhsProps ( staticCast<Dict>( lhs ) );
    Properties  rhsProps ( staticCast<Dict>( rhs ) );

    mergeAndReplace ( lhsProps, rhsProps );

    return lhsProps.getContents ();
  }

  Errors::invalidOperands ( getLineNumber() );

  return nullptr;
}


//-----------------------------------------------------------------------
//   extendArray_
//-----------------------------------------------------------------------


template <class T>

  Ref<Object> Parser::extendArray_

  ( const String&    name,
    const Array<T>&  lhs,
    const ObjRef&    rhs )

{
  const idx_t  n = lhs.size ();

  if ( n == maxOf( n ) )
  {
    Errors::arraySizeOverflow ( getLineNumber() );
  }

  if ( isInstance<T>( rhs ) )
  {
    Array<T>  r ( n + 1 );

    r[slice(BEGIN,n)] = lhs;
    r[n]              = toValue<T> ( rhs );

    return toObject ( r );
  }

  if ( isInstance< Array<T> >( rhs ) )
  {
    const Array<T>&  b = toValue<const Array<T>&> ( rhs );
    Array<T>         r ( n + b.size() );

    r[slice(BEGIN,n)] = lhs;
    r[slice(n,END)]   = b;

    return toObject ( r );
  }

  Errors::invalidOperands ( getLineNumber() );

  return nullptr;
}


//-----------------------------------------------------------------------
//   nextToken_
//-----------------------------------------------------------------------


inline int Parser::nextToken_ ()
{
  int  token = input_->nextToken ();

  if ( token == Tokenizer::SYMBOL_TOKEN )
  {
    return symbolToken_ ();
  }
  else
  {
    return token;
  }
}


//-----------------------------------------------------------------------
//   symbolToken_
//-----------------------------------------------------------------------


int Parser::symbolToken_ ()
{
  const char*  s;
  idx_t        n;


  s = input_->getTokenChars ( n );

  if ( n == 4 && (s[0] == 'n' && s[1] == 'o' &&
                  s[2] == 'n' && s[3] == 'e') )
  {
    return NONE_TOKEN;
  }

  if ( options_ & Properties::PARSE_INCLUDE )
  {
    if ( n == 7 && (s[0] == 'i' && s[1] == 'n' &&
                    s[2] == 'c' && s[3] == 'l' &&
                    s[4] == 'u' && s[5] == 'd' &&
                    s[6] == 'e') )
    {
      return INCLUDE_TOKEN;
    }

    if ( n == 8 && (s[0] == 's' && s[1] == 'i' &&
                    s[2] == 'n' && s[3] == 'c' &&
                    s[4] == 'l' && s[5] == 'u' &&
                    s[6] == 'd' && s[7] == 'e') )
    {
      return  SINCLUDE_TOKEN;
    }
  }

  return Tokenizer::SYMBOL_TOKEN;
}



JEM_END_NAMESPACE( props )
JEM_END_PACKAGE( util )
