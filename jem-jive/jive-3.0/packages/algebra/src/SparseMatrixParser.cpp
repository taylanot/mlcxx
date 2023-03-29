
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
#include <jem/numeric/sparse/utilities.h>
#include <jem/util/SparseArray.h>
#include <jem/util/FastAllocator.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/algebra/SparseMatrixParser.h>


JEM_DEFINE_CLASS( jive::algebra::SparseMatrixParser );


using jem::util::SparseArray;
using jem::util::FastAllocator;
using jem::xml::ParseLog;
using jem::xml::Tokenizer;


JIVE_BEGIN_PACKAGE( algebra )


//=======================================================================
//   class SparseMatrixParser::Utils_
//=======================================================================


class SparseMatrixParser::Utils_
{
 public:

  static void           parse

    ( State&              state,
      SparseMatrixObj&    mat );

  static void           scan

    ( State&              state );

  static idx_t          readCount

    ( State&              state,
      const char*         name );

  static inline void    readElement

    ( State&              state,
      idx_t&              irow,
      idx_t&              jcol,
      double&             value,
      idx_t               rowCount,
      idx_t               colCount );

  static inline void    logProgress

    ( State&              state,
      idx_t&              counter,
      idx_t               irow,
      idx_t               jcol );

  static inline void    checkIndex

    ( State&              state,
      idx_t               index,
      idx_t               ubound,
      const char*         name );

};


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void SparseMatrixParser::Utils_::parse

  ( State&            state,
    SparseMatrixObj&  mat )

{
  typedef SparseArray<double,2>  MBuffer;

  Tokenizer&  input = * state.input;

  MBuffer     buffer;

  double      value;

  idx_t       rowCount;
  idx_t       colCount;
  idx_t       nonZeroCount;
  idx_t       irow;
  idx_t       jcol;
  idx_t       counter;


  rowCount     = Utils_::readCount ( state, "rows" );
  colCount     = Utils_::readCount ( state, "columns" );
  nonZeroCount = Utils_::readCount ( state, "non-zeroes" );
  counter      = 0;

  buffer.reshape ( jem::shape( rowCount, colCount ) );
  buffer.reserve ( nonZeroCount );

  while ( input.nextToken () == Tokenizer::INTEGER_TOKEN )
  {
    counter++;

    readElement ( state, irow, jcol, value, rowCount, colCount );

    buffer(irow - 1, jcol - 1) = value;

    logProgress ( state, counter, irow, jcol );
  }

  input.pushBackToken ();

  mat.setMatrix ( jem::numeric::toMatrix( buffer ) );
}


//-----------------------------------------------------------------------
//   scan
//-----------------------------------------------------------------------


void SparseMatrixParser::Utils_::scan ( State& state )
{
  Tokenizer&  input = * state.input;

  double      value;

  idx_t       rowCount;
  idx_t       colCount;
  idx_t       irow;
  idx_t       jcol;
  idx_t       counter;


  rowCount = Utils_::readCount ( state, "rows" );
  colCount = Utils_::readCount ( state, "columns" );
  counter  = 0;

  while ( input.nextToken () == Tokenizer::INTEGER_TOKEN )
  {
    counter++;

    readElement ( state, irow, jcol, value, rowCount, colCount );
    logProgress ( state, counter, irow, jcol );
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   readCount
//-----------------------------------------------------------------------


idx_t SparseMatrixParser::Utils_::readCount

  ( State&       state,
    const char*  name )

{
  Tokenizer& input = * state.input;

  idx_t  n;


  if ( input.getToken() != Tokenizer::INTEGER_TOKEN )
  {
    parseError (
      state,
      String::format ( "expected the number of %s", name )
    );
  }

  n = (idx_t) input.getInteger ();

  if ( n < 0 )
  {
    inputError (
      state,
      String::format ( "invalid number of %s: %d", name, n )
    );
  }

  return n;
}


//-----------------------------------------------------------------------
//   readElement
//-----------------------------------------------------------------------


inline void SparseMatrixParser::Utils_::readElement

  ( State&   state,
    idx_t&   irow,
    idx_t&   jcol,
    double&  value,
    idx_t    rowCount,
    idx_t    colCount )

{
  Tokenizer&  input = * state.input;

  idx_t  token;


  irow = (idx_t) input.getInteger ();

  checkIndex ( state, irow, rowCount, "row" );

  if ( input.nextToken() != Tokenizer::INTEGER_TOKEN )
  {
    parseError ( state, "expected a column index" );
  }

  jcol = (idx_t) input.getInteger ();

  checkIndex ( state, jcol, colCount, "column" );

  token = input.nextToken ();

  if ( token != Tokenizer::FLOAT_TOKEN &&
       token != Tokenizer::INTEGER_TOKEN )
  {
    parseError ( state, "expected a matrix element" );
  }

  value = input.getFloat ();
}


//-----------------------------------------------------------------------
//   logProgress
//-----------------------------------------------------------------------


inline void SparseMatrixParser::Utils_::logProgress

  ( State&  state,
    idx_t&  counter,
    idx_t   irow,
    idx_t   jcol )

{
  if ( counter > 1000_idx )
  {
    counter = 0;

    state.log->logEvent (
      state,
      ParseLog::PROGRESS,
      String::format ( "at matrix element (%d,%d)", irow, jcol )
    );
  }
}


//-----------------------------------------------------------------------
//   checkIndex
//-----------------------------------------------------------------------


inline void SparseMatrixParser::Utils_::checkIndex

  ( State&       state,
    idx_t        index,
    idx_t        ubound,
    const char*  name )

{
  if ( index < 1 || index > ubound )
  {
    inputError (
      state,
      String::format ( "invalid %s index: %d", name, index )
    );
  }
}


//=======================================================================
//   class SparseMatrixParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SparseMatrixParser::SparseMatrixParser

  ( Ref<SparseMatrixObj> mat ) :

    matrix_ ( mat )

{}


SparseMatrixParser::~SparseMatrixParser ()
{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void SparseMatrixParser::parse ( State& state )
{
  if ( matrix_ )
  {
    Utils_::parse ( state, * matrix_ );
  }
  else
  {
    Utils_::scan ( state );
  }
}


//-----------------------------------------------------------------------
//   setMatrix
//-----------------------------------------------------------------------


void SparseMatrixParser::setMatrix ( Ref<SparseMatrixObj>  mat )
{
  matrix_ = mat;
}


JIVE_END_PACKAGE( algebra )
