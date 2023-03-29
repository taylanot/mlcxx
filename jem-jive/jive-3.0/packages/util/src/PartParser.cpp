
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/ParserScope.h>
#include <jive/util/ParserActions.h>
#include <jive/util/PartParser.h>


JEM_DEFINE_CLASS( jive::util::PartParser );


using jem::maxOf;
using jem::newInstance;
using jem::Error;
using jem::xml::ParseLog;
using jem::xml::Parser;
using jem::xml::ParserState;
using jem::xml::Tokenizer;
using jem::xml::AttributeParser;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class PartScope
//=======================================================================


class PartScope : public ParserScope
{
 public:

  typedef PartScope         Self;
  typedef ParserScope       Scope;


  explicit inline           PartScope

    ( const Ref<Scope>&       parent );

  void                      init

    ( ParserState&            state,
      const String&           partName,
      idx_t                   partID,
      idx_t                   maxParts );

  virtual idx_t             expandID

    ( idx_t                   id )       const override;

  virtual bool              restrictID

    ( idx_t&                  id )       const override;

  virtual String            expandName

    ( const String&           name )     const override;

  virtual bool              restrictName

    ( String&                 name )     const override;


 private:

  Ref<Scope>                parent_;
  String                    partName_;
  idx_t                     partID_;
  idx_t                     maxParts_;
  idx_t                     maxID_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline PartScope::PartScope ( const Ref<Scope>& p ) :

  parent_ ( p )

{
  if ( p->isGlobal() )
  {
    parent_ = nullptr;
  }

  partID_   = 0;
  maxParts_ = 1;
  maxID_    = maxOf ( maxID_ );
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void PartScope::init

  ( ParserState&   state,
    const String&  partName,
    idx_t          partID,
    idx_t          maxParts )

{
  Self*  p = dynamic_cast<Self*> ( parent_.get() );


  if ( p )
  {
    if ( p->maxParts_ > (maxOf( maxParts ) / maxParts) )
    {
      Parser::inputError (
        state,
        "maximum part depth exceeded (integer overflow)"
      );
    }

    parent_   = nullptr;
    partName_ = joinNames ( p->partName_, partName );
    partID_   = p->maxParts_ * partID + p->partID_;
    maxParts_ = p->maxParts_ * maxParts;
  }
  else
  {
    partName_ = partName;
    partID_   = partID;
    maxParts_ = maxParts;
  }

  maxID_ = maxOf( maxParts_ ) / maxParts_;
}


//-----------------------------------------------------------------------
//   expandID
//-----------------------------------------------------------------------


idx_t PartScope::expandID ( idx_t id ) const
{
  idx_t  sign = 1;

  if ( id < 0 )
  {
    id   = -id;
    sign = -1;
  }

  if ( id > maxID_ )
  {
    id *= sign;

    throw ItemIDException (
      JEM_FUNC,
      String::format (
        "item ID (%d) out of range; valid range is [%d,%d]",
        id,
        -maxID_, maxID_
      ),
      id
    );
  }

  id = sign * (maxParts_ * id + partID_);

  if ( ! parent_ )
  {
    return id;
  }
  else
  {
    return parent_->expandID ( id );
  }
}


//-----------------------------------------------------------------------
//   restrictID
//-----------------------------------------------------------------------


bool PartScope::restrictID ( idx_t& id ) const
{
  idx_t  sign = 1;
  idx_t  ii   = id;

  if ( parent_ )
  {
    if ( ! parent_->restrictID( ii ) )
    {
      return false;
    }
  }

  if ( ii < 0 )
  {
    ii   = -ii;
    sign = -1;
  }

  ii = ii / maxParts_;

  if ( (sign * id - ii * maxParts_) == partID_ )
  {
    id = sign * ii;
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   expandName
//-----------------------------------------------------------------------


String PartScope::expandName ( const String& name ) const
{
  if ( ! parent_ )
  {
    return joinNames ( partName_, name );
  }
  else
  {
    return parent_->expandName ( joinNames( partName_, name ) );
  }
}


//-----------------------------------------------------------------------
//   restrictName
//-----------------------------------------------------------------------


bool PartScope::restrictName ( String& name ) const
{
  const idx_t  n = partName_.size ();
  String       s = name;

  if ( parent_ )
  {
    if ( ! parent_->restrictName( s ) )
    {
      return false;
    }
  }

  if ( n == 0 )
  {
    name = s;
    return true;
  }

  if ( s.size() <= n )
  {
    return false;
  }

  if ( s[n] != '.' )
  {
    return false;
  }

  if ( s.find( partName_ ) != 0 )
  {
    return false;
  }

  name = s[slice(n + 1,END)];

  return true;
}


//=======================================================================
//   class PartEndParser
//=======================================================================


class PartEndParser : public jem::xml::Parser
{
 public:

  typedef PartEndParser     Self;
  typedef jem::xml::Parser  Super;


  inline                    PartEndParser

    ( const Ref<PartParser>&  parent,
      const String&           tagName,
      idx_t                   partID );

  virtual void              parse

    ( State&                  state )  override;


 private:

  Ref<PartParser>           parent_;
  Ref<ParserScope>          scope_;

  Properties                parts_;
  String                    tagName_;
  idx_t                     partID_;
  String                    partName_;
  idx_t                     maxParts_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline PartEndParser::PartEndParser

  ( const Ref<PartParser>&  p,
    const String&           tagName,
    idx_t                   partID ) :

    parent_   ( p                    ),
    scope_    ( p->scope_            ),
    parts_    ( p->parts_            ),
    tagName_  ( tagName              ),
    partID_   ( partID               ),
    partName_ ( p->attribs_.partName ),
    maxParts_ ( p->attribs_.maxParts )

{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void PartEndParser::parse ( State& state )
{
  Ref<Parser>  p = state.parserStack.top ();


  parseEndTag ( state, tagName_ );

  state.log->logEvent (
    state,
    ParseLog::OTHER,
    String::format ( "leaving part `%s\' (ID = %d)",
                     partName_,
                     partID_ )
  );

  parent_->scope_    = scope_;
  parent_->parts_    = parts_;
  parent_->maxParts_ = maxParts_;

  if ( p )
  {
    Properties  params ( "parserParams" );

    params.set    ( ParserParams ::SCOPE,     scope_ );
    p->takeAction ( ParserActions::NEW_SCOPE, params );
  }
}


//=======================================================================
//   class PartParser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  PartParser::TAG_NAME          = "Part";
const char*  PartParser::ATTRIB_NAMES_     = "name | maxParts";
const idx_t  PartParser::NAME_ATTRIB_      = 0;
const idx_t  PartParser::MAX_PARTS_ATTRIB_ = 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PartParser::PartParser ( idx_t maxParts ) :

  AttributeParser ( ATTRIB_NAMES_ ),
  parts_          ( "parts"       )

{
  JEM_PRECHECK2 ( maxParts > 0, "invalid number of parts" );

  scope_            = newInstance<GlobalScope> ();
  maxParts_         = maxParts;
  attribs_.maxParts = maxParts;
}


PartParser::~PartParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void PartParser::reset ()
{
  parts_.clear ();
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void PartParser::parse ( State& state )
{
  Ref<Parser>     p     =   state.parserStack.top ();
  Tokenizer&      input = * state.input;
  Attribs_&       a     =   attribs_;

  Ref<PartScope>  pscope;

  String          tagName;
  idx_t           partID;
  int             token;


  if ( input.getToken() != Tokenizer::START_TAG_TOKEN )
  {
    parseError ( state, "current token is not a start tag" );
  }

  a.partName = "";
  a.maxParts = maxParts_;
  tagName    = input.getTagName ();
  token      = parseAttributes  ( state );

  if ( ! parts_.find( partID, a.partName ) )
  {
    partID = parts_.getContents()->size ();

    if ( partID >= maxParts_ )
    {
      inputError (
        state,
        String::format (
          "maximum number of parts (%d) exceeded; "
          "use the attribute `maxParts\' to increase "
          "the maximum",
          maxParts_
        )
      );
    }

    parts_.set ( a.partName, partID );
  }

  if ( token != Tokenizer::TAG_END_TOKEN )
  {
    state.log->logEvent ( state, ParseLog::TAG_START, tagName );
    return;
  }

  state.pushParser (
    newInstance<PartEndParser> ( this, tagName, partID )
  );

  pscope = newInstance<PartScope> ( scope_ );

  pscope->init ( state, a.partName, partID, maxParts_ );

  scope_    = pscope;
  parts_    = Properties ( "parts" );
  maxParts_ = a.maxParts;

  state.log->logEvent (
    state,
    ParseLog::OTHER,
    String::format ( "entering part `%s\' (ID = %d)",
                     a.partName,
                     partID )
  );

  if ( p )
  {
    Properties  params ( "parserParams" );

    params.set       ( ParserParams::SCOPE,      scope_ );
    p->takeAction    ( ParserActions::NEW_SCOPE, params );
    state.pushParser ( p );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void PartParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if ( action == ParserActions::NEW_SCOPE )
  {
    params.get ( scope_, ParserParams::SCOPE );
  }
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void PartParser::handleAttribute

  ( State&         state,
    const String&  name,
    const String&  value,
    idx_t          index )

{
  if      ( index == NAME_ATTRIB_ )
  {
    attribs_.partName = value.stripWhite ();
  }
  else if ( index == MAX_PARTS_ATTRIB_ )
  {
    attribs_.maxParts = (idx_t)

      parseInteger ( state, name, value,
                     1,     maxOf ( attribs_.maxParts ) );
  }
  else
  {
    throw Error ( JEM_FUNC, "unexpected attribute index" );
  }
}


JIVE_END_PACKAGE( util )
