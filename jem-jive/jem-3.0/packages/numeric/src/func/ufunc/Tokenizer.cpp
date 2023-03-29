
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
#include <jem/base/Once.h>
#include <jem/base/ParseException.h>
#include <jem/base/IllegalInputException.h>
#include "Tokenizer.h"


JEM_BEGIN_UFUNC_IMP


//=======================================================================
//   class Tokenizer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


StringMap*  Tokenizer::keywords_ = 0;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


Tokenizer::Tokenizer ( const String& str ) :

  scanner_ ( str )

{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, initOnce_ );

  token_  = pushedToken_ = Tokens::NONE;
  number_ = 0.0;
}


//-----------------------------------------------------------------------
//   nextToken
//-----------------------------------------------------------------------


int Tokenizer::nextToken ()
{
  if ( pushedToken_ != Tokens::NONE )
  {
    token_       = pushedToken_;
    pushedToken_ = Tokens::NONE;
  }
  else
  {
    try
    {
      nextToken_ ();
    }
    catch ( const ParseException& ex )
    {
      throw ParseException (
        ERR_CONTEXT,
        errPrefix + ex.what ()
      );
    }
    catch ( const IllegalInputException& ex )
    {
      throw IllegalInputException (
        ERR_CONTEXT,
        errPrefix + ex.what ()
      );
    }
    catch ( ... )
    {
      throw;
    }
  }

  return token_;
}


//-----------------------------------------------------------------------
//   nextToken_
//-----------------------------------------------------------------------


void Tokenizer::nextToken_ ()
{
  int  c, n;


  scanner_.skipWhite ();

  c = scanner_.read  ();

  if ( c < 0 )
  {
    scanner_.setTokenString ( "end of file" );

    token_ = Tokens::END;
    return;
  }

  token_ = Tokens::NONE;

  scanner_.token.clear ();

  scanner_.token += (char) c;

  if ( (std::isalpha(c) || c == '_') )
  {
    scanner_.readSymbol ();

    symbol_ = scanner_.getTokenString ();

    StringMap::Iterator  it = keywords_->find ( symbol_ );

    if ( it == keywords_->end() )
    {
      token_ = Tokens::SYMBOL;
    }
    else
    {
      token_ = (int) it->second;
    }

    return;
  }

  if ( c == '$' )
  {
    scanner_.readDollar ();

    symbol_ = scanner_.getTokenString ();
    token_  = Tokens::SYMBOL;

    return;
  }

  if ( std::isdigit(c) )
  {
    if ( scanner_.readNumber () == Scanner::INTEGER )
    {
      number_ = (double ) scanner_.getInteger ();
    }
    else
    {
      number_ = scanner_.getFloat ();
    }

    token_ = Tokens::NUMBER;

    return;
  }

  switch ( c )
  {
  case '.':

    n = scanner_.read ();

    if ( std::isdigit(n) )
    {
      scanner_.token += (char) n;

      scanner_.readFloat ();

      number_ = scanner_.getFloat ();
      token_  = Tokens::NUMBER;
    }
    else
    {
      scanner_.unread ( n );

      token_ = c;
    }

    break;

  case '|':

    token_ = nextToken_ ( '|', Tokens::OR,  c );

    break;

  case '&':

    token_ = nextToken_ ( '&', Tokens::AND, c );

    break;

  case '<':

    token_ = nextToken_ ( '=', Tokens::LE,  c );

    break;

  case '>':

    token_ = nextToken_ ( '=', Tokens::GE,  c );

    break;

  default:

    token_ = c;
  }
}


inline int Tokenizer::nextToken_

  ( int   c1,
    int   tok1,
    int   tok2 )

{
  int  next = scanner_.read ();

  if ( next == c1 )
  {
    scanner_.token += (char) next;

    return tok1;
  }
  else
  {
    scanner_.unread ( next );

    return tok2;
  }
}


//-----------------------------------------------------------------------
//   initOnce_
//-----------------------------------------------------------------------


#undef  JEM_ADD_KEYWORD
#define JEM_ADD_KEYWORD( TOK )                  \
                                                \
  keywords_->insert (                           \
    makePair (                                  \
      Tokens::toString ( Tokens::TOK ),         \
      (int) Tokens::TOK                         \
    )                                           \
  )


void Tokenizer::initOnce_ ()
{
  keywords_ = new StringMap ();

  JEM_ADD_KEYWORD( SAVE    );
  JEM_ADD_KEYWORD( LET     );
  JEM_ADD_KEYWORD( RETURN  );
  JEM_ADD_KEYWORD( IF      );
  JEM_ADD_KEYWORD( ELSE    );
  JEM_ADD_KEYWORD( ELIF    );
  JEM_ADD_KEYWORD( ENDIF   );
  JEM_ADD_KEYWORD( PI      );
  JEM_ADD_KEYWORD( EPS     );
  JEM_ADD_KEYWORD( BIGNUM  );
  JEM_ADD_KEYWORD( TINY    );
  JEM_ADD_KEYWORD( TRUE    );
  JEM_ADD_KEYWORD( FALSE   );
  JEM_ADD_KEYWORD( ABS     );
  JEM_ADD_KEYWORD( SQRT    );
  JEM_ADD_KEYWORD( CEIL    );
  JEM_ADD_KEYWORD( FLOOR   );
  JEM_ADD_KEYWORD( POW     );
  JEM_ADD_KEYWORD( EXP     );
  JEM_ADD_KEYWORD( LOG     );
  JEM_ADD_KEYWORD( LOG10   );
  JEM_ADD_KEYWORD( SIN     );
  JEM_ADD_KEYWORD( COS     );
  JEM_ADD_KEYWORD( TAN     );
  JEM_ADD_KEYWORD( ASIN    );
  JEM_ADD_KEYWORD( ACOS    );
  JEM_ADD_KEYWORD( ATAN    );
  JEM_ADD_KEYWORD( MAX     );
  JEM_ADD_KEYWORD( MIN     );
  JEM_ADD_KEYWORD( DEFINED );
}


JEM_END_UFUNC_IMP
