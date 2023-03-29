
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


#include <jem/base/ParseException.h>
#include "Tokenizer.h"
#include "StdArgList.h"


JEM_BEGIN_UFUNC_IMP


//=======================================================================
//   class StdArgList
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


StdArgList::StdArgList ( const String& args )
{
  Tokenizer  tokzer ( args );

  String     arg;
  int        token;


  tokzer.errPrefix = "invalid function arguments: ";

  token = tokzer.nextToken ();

  while ( token == Tokens::SYMBOL )
  {
    arg = tokzer.getSymbol ();

    if ( argMap_.contains( arg ) )
    {
      throw ParseException (
        ERR_CONTEXT,
        String::format (
          "function argument `%s\' declared twice", arg
        )
      );
    }

    argMap_.insert ( makePair( arg, argMap_.size() ) );

    token = tokzer.nextToken ();

    if ( token == ',' )
    {
      token = tokzer.nextToken ();
    }
    else
    {
      break;
    }
  }

  if ( token != Tokens::END )
  {
    throw ParseException (
      ERR_CONTEXT,
      String::format (
        "invalid function arguments: `%s\'", args
      )
    );
  }
}


//-----------------------------------------------------------------------
//   argCount
//-----------------------------------------------------------------------


idx_t StdArgList::argCount () const
{
  return argMap_.size ();
}


//-----------------------------------------------------------------------
//   getArgIndex
//-----------------------------------------------------------------------


idx_t StdArgList::getArgIndex ( const String& arg )
{
  StringMapIter  it = argMap_.find ( arg );

  if ( it == argMap_.end() )
  {
    return -1;
  }
  else
  {
    return it->second;
  }
}


JEM_END_UFUNC_IMP
