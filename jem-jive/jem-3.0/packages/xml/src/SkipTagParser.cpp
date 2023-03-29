
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


#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/SkipTagParser.h>


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class SkipTagParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SkipTagParser::SkipTagParser () noexcept
{}


SkipTagParser::~SkipTagParser ()
{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void SkipTagParser::parse ( State& state )
{
  Tokenizer& input = * state.input;

  String     tagName;
  int        token;


  if ( input.getToken() != Tokenizer::START_TAG_TOKEN )
  {
    parseError ( state, "current token is not a start tag" );
  }

  tagName = input.getTagName ();

  state.log->logEvent ( state,
                        ParseLog::TAG_SKIP,
                        tagName );

  token = input.skipAttributes ();

  if ( token == Tokenizer::TAG_END_TOKEN )
  {
    input.skipToEndTag ( tagName );
  }
}


JEM_END_PACKAGE( xml )
