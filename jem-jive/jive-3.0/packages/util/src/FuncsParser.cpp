
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


#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/numeric/func/Function.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jive/util/ParserScope.h>
#include <jive/util/ParserActions.h>
#include <jive/util/FuncParser.h>
#include <jive/util/FuncsParser.h>


JEM_DEFINE_CLASS( jive::util::FuncsParser );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;
using jem::Error;
using jem::util::Pattern;


//=======================================================================
//   class FuncsParser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  FuncsParser::ATTRIBS_     = "name";
const idx_t  FuncsParser::NAME_ATTRIB_ = 0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


FuncsParser::FuncsParser

  ( const Ref<Dictionary>&  funcs,
    const String&           filter ) :

    Super       ( ATTRIBS_ ),
    funcs_      (    funcs ),
    nameFilter_ (   filter )

{
  funcParser_ = newInstance<FuncParser> ();

  Pattern::check ( filter );
}


FuncsParser::~FuncsParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void FuncsParser::reset ()
{
  Super::reset ();

  funcName_ = "";
}


//-----------------------------------------------------------------------
//   parseContents
//-----------------------------------------------------------------------


void FuncsParser::parseContents ( State& state )
{
  if ( Pattern::matches( nameFilter_, funcName_ ) )
  {
    if ( funcs_ && funcs_->contains( funcName_ ) )
    {
      inputError (
        state,
        String::format (
          "function `%s\' has already been defined", funcName_
        )
      );
    }

    state.pushParser ( funcParser_ );
  }
  else
  {
    state.input->skipToEndTag  ( getTagName() );
    state.input->pushBackToken ();
  }
}


//-----------------------------------------------------------------------
//   handleEmptyContents
//-----------------------------------------------------------------------


void FuncsParser::handleEmptyContents ( State& state )
{
  if ( Pattern::matches( nameFilter_, funcName_ ) )
  {
    inputError ( state, "at least two data pairs must be specified" );
  }
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void FuncsParser::finish ( State& state )
{
  Ref<Function>  func = funcParser_->getFunction ();

  if ( funcs_ && func )
  {
    funcs_->insert ( funcName_, func );
  }

  funcParser_->reset ();

  funcName_ = "";
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void FuncsParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  Super      ::takeAction ( action, params );
  funcParser_->takeAction ( action, params );

  if ( action == ParserActions::NEW_SCOPE )
  {
    params.get ( scope_, ParserParams::SCOPE );

    if ( scope_->isGlobal() )
    {
      scope_ = nullptr;
    }
  }
}


//-----------------------------------------------------------------------
//   setFuncs
//-----------------------------------------------------------------------


void FuncsParser::setFuncs ( const Ref<Dictionary>& funcs )
{
  funcs_ = funcs;
}


//-----------------------------------------------------------------------
//   setNameFilter
//-----------------------------------------------------------------------


void FuncsParser::setNameFilter ( const String& pattern )
{
  Pattern::check ( pattern );

  nameFilter_ = pattern;
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void FuncsParser::handleAttribute

  ( State&         state,
    const String&  name,
    const String&  value,
    idx_t          index )

{
  if ( index == NAME_ATTRIB_ )
  {
    funcName_ = value.stripWhite ();

    if ( funcName_.size() == 0 )
    {
      inputError ( state, "empty name" );
    }

    if ( scope_ )
    {
      funcName_ = scope_->expandName ( funcName_ );
    }
  }
  else
  {
    throw Error (
      JEM_FUNC,
      "invalid attribute index: " + String( index )
    );
  }
}


JIVE_END_PACKAGE( util )
