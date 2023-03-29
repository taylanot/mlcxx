
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

#ifndef JEM_NUMERIC_FUNC_UFUNC_TOKENIZER_H
#define JEM_NUMERIC_FUNC_UFUNC_TOKENIZER_H

#include "Tokens.h"
#include "Scanner.h"
#include "StringMap.h"


JEM_BEGIN_UFUNC_IMP


//-----------------------------------------------------------------------
//   class Tokenizer
//-----------------------------------------------------------------------


class Tokenizer
{
 public:

  explicit                  Tokenizer

    ( const String&           str = "" );

  inline void               setInput

    ( const String&           str );

  inline int                getToken        () const;
  int                       nextToken       ();
  inline void               pushBackToken   ();
  inline String             getTokenString  () const;
  inline String             getSymbol       () const;
  inline double             getNumber       () const;


 public:

  String                    errPrefix;


 private:

  void                      nextToken_      ();

  inline int                nextToken_

    ( int                     c1,
      int                     tok1,
      int                     tok2 );

  static void               initOnce_       ();


 private:

  static StringMap*         keywords_;

  Scanner                   scanner_;
  int                       token_;
  int                       pushedToken_;
  double                    number_;
  String                    symbol_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   setInput
//-----------------------------------------------------------------------


inline void Tokenizer::setInput ( const String& str )
{
  scanner_.setInput ( str );

  token_ = pushedToken_ = Tokens::NONE;
}


//-----------------------------------------------------------------------
//   getToken
//-----------------------------------------------------------------------


inline int Tokenizer::getToken () const
{
  return token_;
}


//-----------------------------------------------------------------------
//   pushBackToken
//-----------------------------------------------------------------------


inline void Tokenizer::pushBackToken ()
{
  pushedToken_ = token_;
}


//-----------------------------------------------------------------------
//   getTokenString
//-----------------------------------------------------------------------


inline String Tokenizer::getTokenString () const
{
  return scanner_.getTokenString ();
}


//-----------------------------------------------------------------------
//   getSymbol
//-----------------------------------------------------------------------


inline String Tokenizer::getSymbol () const
{
  JEM_ASSERT2 ( token_ == Tokens::SYMBOL,
                "current token is not a symbol" );

  return symbol_;
}


//-----------------------------------------------------------------------
//   getNumber
//-----------------------------------------------------------------------


inline double Tokenizer::getNumber () const
{
  JEM_ASSERT2 ( token_ == Tokens::NUMBER,
                "current token is not a number" );

  return number_;
}


JEM_END_UFUNC_IMP

#endif
