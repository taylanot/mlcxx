
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

#ifndef JIVE_UTIL_PARSEUTILS_H
#define JIVE_UTIL_PARSEUTILS_H

#include <jem/xml/Tokenizer.h>
#include <jive/util/ParserScope.h>


namespace jem
{
  namespace xml
  {
    class ParserState;
  }
}


JIVE_BEGIN_PACKAGE( util )


class ItemSet;
class ItemIDException;


//-----------------------------------------------------------------------
//   class ParseUtils
//-----------------------------------------------------------------------


class ParseUtils
{
 public:

  typedef jem::
    xml::ParserState        State;
  typedef jem::
    xml::Tokenizer          Tokenizer;

  typedef ParserScope       Scope;


  static inline idx_t       getItemID

    ( Tokenizer&              input,
      const Scope*            scope );

  static inline String      getSymbol

    ( Tokenizer&              input,
      const Scope*            scope );

  static inline String      getString

    ( Tokenizer&              input,
      const Scope*            scope  );

  static void               parseRange

    ( State&                  state,
      idx_t&                  first,
      idx_t&                  last,
      idx_t&                  stride );

  static void               itemIDError

    ( State&                  state,
      const ItemSet&          items,
      const ItemIDException&  ex );

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getItemID
//-----------------------------------------------------------------------


inline idx_t ParseUtils::getItemID

  ( Tokenizer&    input,
    const Scope*  scope )

{
  if ( scope )
  {
    return scope->expandID ( (idx_t) input.getInteger() );
  }
  else
  {
    return (idx_t) input.getInteger ();
  }
}


//-----------------------------------------------------------------------
//   getSymbol
//-----------------------------------------------------------------------


inline String ParseUtils::getSymbol

  ( Tokenizer&    input,
    const Scope*  scope )

{
  if ( scope )
  {
    return scope->expandName ( input.getSymbol() );
  }
  else
  {
    return input.getSymbol ();
  }
}


//-----------------------------------------------------------------------
//   getString
//-----------------------------------------------------------------------


inline String ParseUtils::getString

  ( Tokenizer&    input,
    const Scope*  scope )

{
  if ( scope )
  {
    return scope->expandName ( input.getString() );
  }
  else
  {
    return input.getString ();
  }
}


JIVE_END_PACKAGE( util )

#endif
