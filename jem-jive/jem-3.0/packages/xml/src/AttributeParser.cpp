
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
#include <jem/base/Boolean.h>
#include <jem/base/Integer.h>
#include <jem/base/Float.h>
#include <jem/xml/Parser.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/AttributeParser.h>


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class AttributeParser::Attrib_
//=======================================================================


class AttributeParser::Attrib_
{
 public:

  inline     Attrib_ ( const String& name, Kind kind );


  Attrib_*   next;
  String     name;
  Kind       kind;
  bool       isSet;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline AttributeParser::Attrib_::Attrib_

  ( const String&  n,
    Kind           k ) :

    next  ( nullptr ),
    name  ( n ),
    kind  ( k ),
    isSet ( false )

{}


//=======================================================================
//   class AttributeParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


AttributeParser::AttributeParser () :

  list_ ( nullptr )

{}


AttributeParser::AttributeParser ( const String& attribs ) :

  list_ ( nullptr )

{
  try
  {
    declareAttributes ( attribs );
  }
  catch ( ... )
  {
    clear ();
    throw;
  }
}



AttributeParser::~AttributeParser ()
{
  clear ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void AttributeParser::clear ()
{
  Attrib_*  next;

  while ( list_ )
  {
    next  = list_->next;
    delete list_;
    list_ = next;
  }

  list_ = nullptr;
}


//-----------------------------------------------------------------------
//   declareAttribute
//-----------------------------------------------------------------------


idx_t AttributeParser::declareAttribute

  ( const String&  name,
    Kind           kind )

{
  Attrib_*  attrib = list_;
  Attrib_*  prev   = list_;
  idx_t     index  = 0;


  while ( attrib )
  {
    if ( attrib->name == name )
    {
      attrib->kind = kind;
      return index;
    }

    index++;
    prev   = attrib;
    attrib = attrib->next;
  }

  if ( prev )
  {
    prev->next = new Attrib_ ( name, kind );
  }
  else
  {
    list_      = new Attrib_ ( name, kind );
  }

  return index;
}


//-----------------------------------------------------------------------
//   declareAttributes
//-----------------------------------------------------------------------


idx_t AttributeParser::declareAttributes ( const String& attribs )
{
  const idx_t  n = attribs.size ();

  Attrib_*     tail;
  Kind         kind;
  idx_t        count;
  idx_t        i, j;


  clear ();

  kind  = REQUIRED;
  count = 0;
  tail  = nullptr;
  i     = 0;
  j     = 0;

  while ( j < n )
  {
    for ( i = j; i < n &&   std::isspace( attribs[i] ); i++ ) ;
    for ( j = i; j < n && ! std::isspace( attribs[j] ); j++ ) ;

    if ( j == i + 1 && attribs[i] == '|' )
    {
      kind = OPTIONAL;
    }
    else if ( j > i )
    {
      if ( tail )
      {
        tail->next = new Attrib_ ( attribs[slice(i,j)], kind );
        tail       = tail->next;
      }
      else
      {
        list_ = tail = new Attrib_ ( attribs[slice(i,j)], kind );
      }

      count++;
    }
  }

  return count;
}


//-----------------------------------------------------------------------
//   parseAttributes
//-----------------------------------------------------------------------


int AttributeParser::parseAttributes

  ( ParserState&  state,
    bool          lenient )

{
  Tokenizer&  input = *state.input;

  Attrib_*    attrib;
  String      name;
  String      value;
  idx_t       index;


  for ( attrib = list_; attrib; attrib = attrib->next )
  {
    attrib->isSet = false;
  }

  while ( input.readAttribute ( name, value ) )
  {
    index  = 0;

    for ( attrib = list_; attrib; attrib = attrib->next )
    {
      if ( name.equalsIgnoreCase( attrib->name ) )
      {
        attrib->isSet = true;
        handleAttribute ( state, name, value, index );
        break;
      }

      index++;
    }

    if ( ! attrib && ! lenient )
    {
      Parser::illegalAttributeError ( state, name );
    }
  }

  for ( attrib = list_; attrib; attrib = attrib->next )
  {
    if ( ! attrib->isSet && attrib->kind == REQUIRED )
    {
      Parser::missingAttributeError ( state, attrib->name );
    }
  }

  return input.getToken ();
}


//-----------------------------------------------------------------------
//   parseBoolean
//-----------------------------------------------------------------------


bool AttributeParser::parseBoolean

  ( ParserState&   state,
    const String&  name,
    const String&  value )

{
  const char*  s = value.addr ();
  const idx_t  n = value.size ();

  bool  b;

  if      ( n == 0 )
  {
    b = true;
  }
  else if ( n == 1 && s[0] == '1' )
  {
    b = true;
  }
  else if ( n == 1 && s[0] == '0' )
  {
    b = false;
  }
  else if ( n == 2 && (tolower(s[0]) == 'n' &&
                       tolower(s[1]) == 'o') )
  {
    b = false;
  }
  else if ( n == 3 && (tolower(s[0]) == 'y' &&
                       tolower(s[1]) == 'e' &&
                       tolower(s[2]) == 's') )
  {
    b = true;
  }
  else if ( n == 4 && (tolower(s[0]) == 't' &&
                       tolower(s[1]) == 'r' &&
                       tolower(s[2]) == 'u' &&
                       tolower(s[3]) == 'e') )
  {
    b = true;
  }
  else if ( n == 5 && (tolower(s[0]) == 'f' &&
                       tolower(s[1]) == 'a' &&
                       tolower(s[2]) == 'l' &&
                       tolower(s[3]) == 's' &&
                       tolower(s[4]) == 'e') )
  {
    b = false;
  }
  else
  {
    b = false;

    Parser::parseError (
      state,
      String::format (
        "attribute `%s\' is not a valid boolean", name
      )
    );
  }

  return b;
}


//-----------------------------------------------------------------------
//   parseInteger
//-----------------------------------------------------------------------


lint AttributeParser::parseInteger

  ( ParserState&   state,
    const String&  name,
    const String&  value )

{
  lint  i;

  if ( ! Integer::parse ( i, value ) )
  {
    Parser::parseError (
      state,
      String::format (
        "attribute `%s\' is not a valid integer", name
      )
    );
  }

  return i;
}


lint AttributeParser::parseInteger

  ( ParserState&   state,
    const String&  name,
    const String&  value,
    lint           min,
    lint           max )

{
  lint  i = parseInteger ( state, name, value );

  if ( i < min || i > max )
  {
    Parser::parseError (
      state,
      String::format (
        "attribute `%s\' out of range; valid range "
        "is [%d:%d]", name, min, max
      )
    );
  }

  return i;
}


//-----------------------------------------------------------------------
//   parseFloat
//-----------------------------------------------------------------------


double AttributeParser::parseFloat

  ( ParserState&   state,
    const String&  name,
    const String&  value )

{
  double  d;

  if ( ! Float::parse ( d, value ) )
  {
    Parser::parseError (
      state,
      String::format (
        "attribute `%s\' is not a valid float", name
      )
    );
  }

  return d;
}


double AttributeParser::parseFloat

  ( ParserState&   state,
    const String&  name,
    const String&  value,
    double         min,
    double         max )

{
  double  d = parseFloat ( state, name, value );

  if ( d < min || d > max )
  {
    Parser::parseError (
      state,
      String::format (
        "attribute `%s\' out of range; valid range "
        "is [%g:%g]", name, min, max
      )
    );
  }

  return d;
}


JEM_END_PACKAGE( xml )
