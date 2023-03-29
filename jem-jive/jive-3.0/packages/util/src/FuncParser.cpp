
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


#include <jem/base/ClassTemplate.h>
#include <jem/numeric/func/DataFunc.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jive/Array.h>
#include <jive/util/FuncParser.h>


JEM_DEFINE_CLASS( jive::util::FuncParser );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;
using jem::util::ArrayBuffer;
using jem::numeric::DataFunc;
using jem::xml::ParseLog;
using jem::xml::Tokenizer;


//=======================================================================
//   class FuncParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FuncParser::FuncParser ()
{}


FuncParser::~FuncParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void FuncParser::reset ()
{
  func_ = nullptr;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void FuncParser::parse ( State& state )
{
  typedef ArrayBuffer<double>  Buffer;

  Tokenizer&  input = * state.input;

  Vector      xData;
  Vector      yData;

  Buffer      buf;

  double      x, x0;
  int         token;
  idx_t       i, n;


  buf.reserve ( 32 );

  func_ = nullptr;
  token = input.nextToken ();
  x     = 0.0;
  n     = 0;

  while ( token == Tokenizer::FLOAT_TOKEN ||
          token == Tokenizer::INTEGER_TOKEN )
  {
    x0 = x;
    x  = input.getFloat ();

    if ( n > 0 && (x - x0) <= 0.0 )
    {
      inputError (
        state,
        "x-data values must be specified in strictly ascending order"
      );
    }

    buf.pushBack ( x );

    token = input.nextToken ();

    if ( token != Tokenizer::FLOAT_TOKEN &&
         token != Tokenizer::INTEGER_TOKEN )
    {
      parseError ( state );
    }

    buf.pushBack ( input.getFloat() );

    token = input.nextToken ();

    if ( token == ';' )
    {
      token = input.nextToken ();
    }

    n++;

    if ( n % 2000_idx == 0 )
    {
      state.log->logEvent (
        state,
        ParseLog::PROGRESS,
        String::format ( "number of data pairs read : %d", n )
      );
    }
  }

  input.pushBackToken ();

  if ( n < 2 )
  {
    inputError ( state,
                 "at least two data pairs must be specified" );
  }

  xData.resize ( n );
  yData.resize ( n );

  for ( i = 0; i < n; i++ )
  {
    xData[i] = buf[2 * i + 0];
    yData[i] = buf[2 * i + 1];
  }

  func_ = newInstance<DataFunc> ( xData, yData );
}


//-----------------------------------------------------------------------
//   getFunction
//-----------------------------------------------------------------------


Ref<Function> FuncParser::getFunction () const
{
  return func_;
}


JIVE_END_PACKAGE( util )
