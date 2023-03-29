
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
#include <jem/util/Properties.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/XPointSet.h>
#include <jive/util/ParseUtils.h>
#include <jive/util/ParserActions.h>
#include <jive/util/PointSetParser.h>


JEM_DEFINE_CLASS( jive::util::PointSetParser );


using jem::util::ArrayBuffer;
using jem::xml::ParseLog;
using jem::xml::Tokenizer;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class PointSetParser::Utils_
//=======================================================================


class PointSetParser::Utils_
{
 public:

  static Vector         parseCoords

    ( State&              state );

  static inline void    parseCoords

    ( State&              state,
      const Vector&       coords );

  static inline void    logProgress

    ( State&              state,
      idx_t&              counter,
      const String&       pointName,
      idx_t               pointID );

};


//-----------------------------------------------------------------------
//   parseCoords (rank unknown)
//-----------------------------------------------------------------------


Vector PointSetParser::Utils_::parseCoords ( State& state )
{
  Tokenizer&           input = * state.input;

  ArrayBuffer<double>  buffer;
  int                  token = 0;


  buffer.reserve ( 8 );

  while ( true )
  {
    token = input.nextToken ();

    if      ( token == Tokenizer::INTEGER_TOKEN )
    {
      buffer.pushBack ( (double) input.getInteger() );
    }
    else if ( token == Tokenizer::FLOAT_TOKEN )
    {
      buffer.pushBack ( input.getFloat() );
    }
    else
    {
      break;
    }
  }

  if ( token != ';' )
  {
    parseError ( state, "expected a `;\'" );
  }

  return buffer.toArray ();
}


//-----------------------------------------------------------------------
//   parseCoords (with known rank)
//-----------------------------------------------------------------------


inline void PointSetParser::Utils_::parseCoords

  ( State&         state,
    const Vector&  coords )

{
  Tokenizer&   input = * state.input;
  const idx_t  rank  = coords.size ();

  int          token;


  for ( idx_t i = 0; i < rank; i++ )
  {
    token = input.nextToken();

    if      ( token == Tokenizer::INTEGER_TOKEN )
    {
      coords[i] = (double) input.getInteger ();
    }
    else if ( token == Tokenizer::FLOAT_TOKEN )
    {
      coords[i] = input.getFloat ();
    }
    else
    {
      parseError ( state, "expected a floating point number" );
    }

  }

  if ( input.nextToken() != ';' )
  {
    parseError ( state, "expected a `;\'" );
  }
}


//-----------------------------------------------------------------------
//   logProgress
//-----------------------------------------------------------------------


inline void PointSetParser::Utils_::logProgress

  ( State&         state,
    idx_t&         counter,
    const String&  pointName,
    idx_t          pointID )

{
  if ( counter > 1000_idx )
  {
    counter = 0;

    state.log->logEvent (
      state,
      ParseLog::PROGRESS,
      String::format ( "at %s %d", pointName, pointID )
    );
  }
}


//=======================================================================
//   class PointSetParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PointSetParser::PointSetParser

  ( const Ref<XPointSet>&  points,
    idx_t                  rank ) :

    points_ ( points ),
    rank_   (   rank )

{}


PointSetParser::~PointSetParser ()
{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void PointSetParser::parse ( State& state )
{
  if ( ! points_ )
  {
    try
    {
      scan_ ( state );
    }
    catch ( const ItemIDException& ex )
    {
      inputError ( state, ex.what() );
    }
  }
  else
  {
    try
    {
      parse_ ( state );
    }
    catch ( const ItemIDException& ex )
    {
      ParseUtils::itemIDError ( state, *points_, ex );
    }
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void PointSetParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
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
//   setPointSet
//-----------------------------------------------------------------------


void PointSetParser::setPointSet ( const Ref<XPointSet>& points )
{
  points_ = points;
}


//-----------------------------------------------------------------------
//   setRank
//-----------------------------------------------------------------------


void PointSetParser::setRank ( idx_t rank )
{
  rank_ = rank;
}


//-----------------------------------------------------------------------
//   parse_
//-----------------------------------------------------------------------


void PointSetParser::parse_ ( State& state )
{
  XPointSet&    points    = * points_;
  const String  pointName =   points.getItemName  ();
  ParserScope*  scope     =   scope_.get          ();
  Tokenizer&    input     = * state.input;

  Vector        coords;
  int           token;
  idx_t         rank;
  idx_t         pointID;
  idx_t         counter;


  points.reserve ( 256 );

  token = input.nextToken ();

  if ( points.size() == 0 &&
       rank_         <  0 &&
       token         == Tokenizer::INTEGER_TOKEN )
  {
    pointID = ParseUtils::getItemID ( input, scope );

    points.addPoint ( pointID, Utils_::parseCoords( state ) );

    token = input.nextToken ();
  }

  if ( points.size() == 0 && rank_ >= 0 )
  {
    rank = rank_;
  }
  else
  {
    rank = points.rank ();
  }

  counter = 0;
  coords.resize ( rank );

  while ( token == Tokenizer::INTEGER_TOKEN )
  {
    counter++;

    pointID = ParseUtils::getItemID ( input, scope );

    Utils_::parseCoords ( state,   coords );
    points.addPoint     ( pointID, coords );
    Utils_::logProgress ( state,   counter, pointName, pointID );

    token = input.nextToken ();
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   scan_
//-----------------------------------------------------------------------


void PointSetParser::scan_ ( State& state )
{
  const String  pointName =   "point";
  ParserScope*  scope     =   scope_.get ();
  Tokenizer&    input     = * state.input;

  Vector        coords;
  int           token;
  idx_t         pointID;
  idx_t         counter;


  token = input.nextToken ();

  if ( rank_ < 0 && token == Tokenizer::INTEGER_TOKEN )
  {
    pointID = ParseUtils::getItemID ( input, scope );

    coords.ref ( Utils_::parseCoords ( state ) );

    token   = input.nextToken ();
  }
  else if ( rank_ >= 0 )
  {
    coords.resize ( rank_ );
  }

  counter = 0;

  while ( token == Tokenizer::INTEGER_TOKEN )
  {
    counter++;

    pointID = ParseUtils::getItemID ( input, scope );

    Utils_::parseCoords ( state, coords );
    Utils_::logProgress ( state, counter, pointName, pointID );

    token = input.nextToken ();
  }

  input.pushBackToken ();
}


JIVE_END_PACKAGE( util )
