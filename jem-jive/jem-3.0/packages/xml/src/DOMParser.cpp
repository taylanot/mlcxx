
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


#include <jem/base/rtti.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ParseException.h>
#include <jem/util/Flex.h>
#include <jem/util/ObjFlex.h>
#include <jem/util/Tokenizer.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/ParserActions.h>
#include <jem/xml/DOMAttribs.h>
#include <jem/xml/DOMElement.h>
#include <jem/xml/DOMParser.h>


JEM_BEGIN_PACKAGE( xml )


using jem::util::Flex;
using jem::util::ObjFlex;
using jem::util::Properties;

typedef util::Tokenizer  AttribTokenizer;


//=======================================================================
//   class DOMParser::Work_
//=======================================================================


class DOMParser::Work_
{
 public:

  Flex<String>              attribs;
  Ref<AttribTokenizer>      tokenizer;

};


//=======================================================================
//   class DOMParser::Element_
//=======================================================================


class DOMParser::Element_ : public Collectable
{
 public:

  typedef Element_          Self;
  typedef Collectable       Super;

                            Element_    ()       noexcept;

                            Element_

    ( const Ref<Element_>&    parent,
      const String&           typeName );

  inline bool               isRoot      () const noexcept;
  inline bool               isLeaf      () const noexcept;

  inline void               addTextElement

    ( const String&           text );

  Ref<Element_>             finish

    ( const String&           text );

  inline Array
    < Ref<DOMElement> >     getChildren () const;


 public:

  String                    typeName;
  Properties                attribs;


 private:

  Ref<Element_>             parent_;
  ObjFlex                   children_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


DOMParser::Element_::Element_ () noexcept :

  typeName ( "root" ),
  attribs  ( "root" )

{}


DOMParser::Element_::Element_

  ( const Ref<Element_>&  parent,
    const String&         typeName ) :

    typeName ( typeName ),
    attribs  ( typeName ),
    parent_  (   parent )

{}


//-----------------------------------------------------------------------
//   isRoot
//-----------------------------------------------------------------------


inline bool DOMParser::Element_::isRoot () const noexcept
{
  return (parent_ == nullptr);
}


//-----------------------------------------------------------------------
//   isLeaf
//-----------------------------------------------------------------------


inline bool DOMParser::Element_::isLeaf () const noexcept
{
  return (children_.size() == 0);
}


//-----------------------------------------------------------------------
//   addTextElement
//-----------------------------------------------------------------------


inline void DOMParser::Element_::addTextElement

  ( const String&  text )

{
  children_.pushBack (
    newInstance<DOMElement> ( DOMElement::TEXT_TYPE_NAME, text )
  );
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


Ref<DOMParser::Element_> DOMParser::Element_::finish

  ( const String&  text )

{
  Ref<Element_> p = parent_;

  parent_ = nullptr;

  if ( text.size() > 0 )
  {
    if ( isLeaf() )
    {
      attribs.set    ( DOMAttribs::CONTENTS, text );
    }
    else
    {
      addTextElement ( text );
    }
  }

  if ( p )
  {
    p->children_.pushBack (
      newInstance<DOMElement> ( typeName, attribs, getChildren() )
    );
  }

  return p;
}


//-----------------------------------------------------------------------
//   getChildren
//-----------------------------------------------------------------------


inline Array< Ref<DOMElement> >

  DOMParser::Element_::getChildren () const

{
  Array< Ref<DOMElement> >  list ( children_.size() );

  idx_t  i;

  for ( i = 0; i < list.size(); i++ )
  {
    list[i] = children_.getAs<DOMElement> ( i );
  }

  return list;
}


//=======================================================================
//   class DOMParser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DOMParser::PARSE_ATTRIBS   = "domParseAttribs";
const char*  DOMParser::PARSE_TO_UPPER  = "domParseToUpper";
const char*  DOMParser::PARSE_TO_LOWER  = "domParseToLower";
const char*  DOMParser::SKIP_CONTENTS   = "domSkipContents";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DOMParser::DOMParser () noexcept
{
  options_ = 0;
}


DOMParser::~DOMParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void DOMParser::reset ()
{
  root_ = nullptr;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void DOMParser::parse ( State& state )
{
  Tokenizer&     input = * state.input;

  Ref<Element_>  elem;
  StringBuffer   cbuf;
  Work_          work;
  String         text;
  String         tag;
  int            token;


  root_ = nullptr;
  elem  = newInstance<Element_> ();

  elem->attribs.set ( DOMAttribs::SOURCE,
                      state.sourceStack.top() );

  while ( true )
  {
    cbuf.clear ();

    token = input.readText ( cbuf );

    if ( token == Tokenizer::EOF_TOKEN )
    {
      if ( ! elem->isRoot() )
      {
        parseError ( state, "unexpected end of file" );
      }

      break;
    }

    if ( ! (options_ & SKIP_CONTENTS_) )
    {
      text = cbuf.toString ();
    }

    if ( token == Tokenizer::END_TAG_TOKEN )
    {
      if ( elem->isRoot() )
      {
        break;
      }

      tag = input.getTagName ();

      if ( ! tag.equalsIgnoreCase( elem->typeName ) )
      {
        parseError (
          state,
          String::format (
            "expected end tag `%s\'", elem->typeName
          )
        );
      }

      state.log->logEvent ( state, ParseLog::TAG_END, tag );

      elem = elem->finish ( text );

      continue;
    }
    else if ( token == Tokenizer::START_TAG_TOKEN )
    {
      if ( text.size() > 0 )
      {
        elem->addTextElement ( text );
      }

      elem  = newElement_    ( state, work, elem );
      token = input.getToken ();

      if ( token == Tokenizer::EMPTY_TAG_END_TOKEN )
      {
        elem = elem->finish ( "" );
      }
    }
  }

  elem->finish ( text );

  root_ = newInstance<DOMElement> (
    elem->typeName,
    elem->attribs,
    elem->getChildren ()
  );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void DOMParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if      ( action == ParserActions::SET_OPTIONS )
  {
    setOption_ ( PARSE_ATTRIBS,  PARSE_ATTRIBS_,  params );
    setOption_ ( PARSE_TO_UPPER, PARSE_TO_UPPER_, params );
    setOption_ ( PARSE_TO_LOWER, PARSE_TO_LOWER_, params );
    setOption_ ( SKIP_CONTENTS,  SKIP_CONTENTS_,  params );
  }
  else if ( action == ParserActions::RESET_OPTIONS )
  {
    options_ = 0;
  }
}


//-----------------------------------------------------------------------
//   getRootElement
//-----------------------------------------------------------------------


Ref<DOMElement> DOMParser::getRootElement () noexcept
{
  Ref<DOMElement>  root;

  root.swap ( root_ );

  return root;
}


//-----------------------------------------------------------------------
//   setOption_
//-----------------------------------------------------------------------


void DOMParser::setOption_

  ( const char*        name,
    Option_            option,
    const Properties&  params )

{
  bool  opt;

  if ( params.find( opt, name ) )
  {
    if ( opt )
    {
      options_ |=  option;
    }
    else
    {
      options_ &= ~option;
    }
  }
}


//-----------------------------------------------------------------------
//   newElement_
//-----------------------------------------------------------------------


Ref<DOMParser::Element_> DOMParser::newElement_

  ( State&                state,
    Work_&                work,
    const Ref<Element_>&  parent )

{
  String         tag  = state.input->getTagName ();
  Ref<Element_>  elem = newInstance<Element_> ( parent, tag );


  state.log->logEvent ( state, ParseLog::TAG_START, tag );

  elem->attribs.set   ( DOMAttribs::SOURCE,
                        state.sourceStack.top() );

  elem->attribs.set   ( DOMAttribs::LINENO,
                        state.input->getLineNumber() );

  work.attribs.clear  ();

  if ( options_ & PARSE_ATTRIBS_ )
  {
    parseAttribs_     ( state, work, *elem );
  }
  else
  {
    readAttribs_      ( state, work, *elem );
  }

  elem->attribs.set   (
    DOMAttribs::ATTRIB_NAMES,
    Array<String> ( work.attribs.begin(), work.attribs.end() )
  );

  return elem;
}


//-----------------------------------------------------------------------
//   readAttribs_
//-----------------------------------------------------------------------


void DOMParser::readAttribs_

  ( State&     state,
    Work_&     work,
    Element_&  elem )

{
  String  name;
  String  value;

  while ( state.input->readAttribute( name, value ) )
  {
    if ( options_ & PARSE_TO_UPPER_ )
    {
      name = name.toUpper ();
    }
    else if ( options_ & PARSE_TO_LOWER_ )
    {
      name = name.toLower ();
    }

    work.attribs.pushBack ( name );
    elem.attribs.set      ( name, value );
  }
}


//-----------------------------------------------------------------------
//   parseAttribs_
//-----------------------------------------------------------------------


void DOMParser::parseAttribs_

  ( State&     state,
    Work_&     work,
    Element_&  elem )

{
  Ref<Object>  obj;
  String       name;
  String       value;


  if ( ! work.tokenizer )
  {
    work.tokenizer = newInstance<AttribTokenizer> ();
  }

  while ( state.input->readAttribute( name, value ) )
  {
    try
    {
      obj = parseAttrib_ ( work, value );
    }
    catch ( const ParseException& ex )
    {
      String err = ex.what ();

      parseError (
        state,
        String::format (
          "error parsing attribute `%s\': %s", name, err
        )
      );
    }

    if      ( options_ & PARSE_TO_UPPER_ )
    {
      name = name.toUpper ();
    }
    else if ( options_ & PARSE_TO_LOWER_ )
    {
      name = name.toLower ();
    }

    work.attribs.pushBack ( name );
    elem.attribs.set      ( name, obj );
  }
}


//-----------------------------------------------------------------------
//   parseAttrib_
//-----------------------------------------------------------------------


Ref<Object> DOMParser::parseAttrib_

  ( Work_&         work,
    const String&  value )

{
  AttribTokenizer&  tokzer = * work.tokenizer;

  Ref<Object>       obj;
  int               token;


  tokzer.setInput ( value );

  token = tokzer.nextToken ();

  if      ( token == AttribTokenizer::BOOLEAN_TOKEN )
  {
    obj = toObject ( tokzer.getBoolean() );
  }
  else if ( token == AttribTokenizer::INTEGER_TOKEN )
  {
    obj = toObject ( tokzer.getInteger() );
  }
  else if ( token == AttribTokenizer::FLOAT_TOKEN )
  {
    obj = toObject ( tokzer.getFloat() );
  }
  else if ( token == AttribTokenizer::STRING_TOKEN )
  {
    obj = toObject ( tokzer.getString() );
  }
  else
  {
    return toObject ( value );
  }

  if ( tokzer.nextToken() != AttribTokenizer::EOF_TOKEN )
  {
    return toObject ( value );
  }
  else
  {
    return obj;
  }
}


JEM_END_PACKAGE( xml )
