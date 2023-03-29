
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


#include <jem/base/assert.h>
#include <jem/base/Error.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/SimpleTagParser.h>


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class SimpleTagParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SimpleTagParser::SimpleTagParser

  ( const Ref<Parser>&  parser ) :

    parser_ ( parser )

{
  JEM_PRECHECK ( parser );
}


SimpleTagParser::~SimpleTagParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void SimpleTagParser::reset ()
{
  Super  ::reset ();
  parser_->reset ();
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void SimpleTagParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  parser_->takeAction ( action, params );
  Super  ::takeAction ( action, params );
}


//-----------------------------------------------------------------------
//   parseContents
//-----------------------------------------------------------------------


void SimpleTagParser::parseContents ( State& state )
{
  state.pushParser ( parser_ );
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void SimpleTagParser:: handleAttribute

  ( ParserState&,
    const String&  name,
    const String&,
    idx_t )

{
  throw Error (
    JEM_FUNC,
    String::format ( "illegal attribute: `%s\'", name )
  );
}


JEM_END_PACKAGE( xml )
