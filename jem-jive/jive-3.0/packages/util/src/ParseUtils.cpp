
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


#include <jem/util/StringUtils.h>
#include <jem/xml/Parser.h>
#include <jem/xml/ParserState.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/ParseUtils.h>


JIVE_BEGIN_PACKAGE( util )


using jem::util::StringUtils;
using jem::xml::Parser;
using jem::xml::ParserState;


//=======================================================================
//   class ParseUtils
//=======================================================================


void ParseUtils::parseRange

  ( State&  state,
    idx_t&    first,
    idx_t&    last,
    idx_t&    stride )

{
  Tokenizer&  input = *state.input;
  int         token;


  if ( input.nextToken() != Tokenizer::INTEGER_TOKEN )
  {
    Parser::parseError (
      state,
      "invalid range: missing start value"
    );
  }

  first = (idx_t) input.getInteger ();
  token =         input.nextToken  ();

  if ( token == ']' )
  {
    last   = first + 1;
    stride = 1;
    return;
  }

  if ( token != ':' )
  {
    Parser::parseError (
      state,
      "invalid range: missing `:\' after start value"
    );
  }

  if ( input.nextToken() != Tokenizer::INTEGER_TOKEN )
  {
    Parser::parseError (
      state,
      "invalid range: missing end value"
    );
  }

  last = (idx_t) input.getInteger ();

  if ( last < first )
  {
    Parser::parseError (
      state,
      "invalid range: the end value is invalid"
    );
  }

  token = input.nextToken ();

  if ( token == ']' )
  {
    stride = 1;
    return;
  }

  if ( token != ';' )
  {
    Parser::parseError (
      state,
      "invalid range: expected a `]\' or `;\'"
    );
  }

  if ( input.nextToken() != Tokenizer::INTEGER_TOKEN )
  {
    Parser::parseError (
      state,
      "invalid range: missing stride"
    );
  }

  stride = (idx_t) input.getInteger();

  if ( stride <= 0 )
  {
    Parser::parseError (
      state,
      "invalid range: non-positive stride"
    );
  }

  if ( input.nextToken() != ']' )
  {
    Parser::parseError (
      state,
      "invalid range: missing `]\'"
    );
  }
}


//-----------------------------------------------------------------------
//   itemIDError
//-----------------------------------------------------------------------


void ParseUtils::itemIDError

  ( State&                  state,
    const ItemSet&          items,
    const ItemIDException&  ex )

{
  const String  itemName = items.getItemName ();

  Parser::inputError (
    state,
    StringUtils::replace ( ex.what(), "item", itemName )
  );
}


JIVE_END_PACKAGE( util )
