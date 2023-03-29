
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
#include <jem/base/assert.h>
#include <jem/util/Pattern.h>
#include <jem/util/StringUtils.h>
#include <jem/io/NullWriter.h>
#include <jem/io/PatternLogger.h>


using jem::util::Pattern;


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class PatternLogger
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const byte  PatternLogger::PASS_  = 0;
const byte  PatternLogger::BLOCK_ = 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PatternLogger::PatternLogger

  ( const Ref<Writer>&  output,
    const String&       pattern )

{
  JEM_PRECHECK ( output );

  output_[PASS_] = output;

  init_ ( util::StringUtils::split( pattern, '|' ) );
}


PatternLogger::PatternLogger

  ( const Ref<Writer>&    output,
    const Array<String>&  patterns )

{
  JEM_PRECHECK ( output );

  output_[PASS_] = output;

  init_ ( patterns );
}


PatternLogger::PatternLogger ( const Self& rhs ) :

  patterns_ ( rhs.patterns_.clone() ),
  actions_  ( rhs.actions_ .clone() )

{
  for ( int i = 0; i < 2; i++ )
  {
    output_[i] = rhs.output_[i]->dup ();
  }

  lastBlocker_ = rhs.lastBlocker_;
  prevOutput_  = output_[BLOCK_].get ();
}


PatternLogger::~PatternLogger ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Logger> PatternLogger::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void PatternLogger::flush ()
{
  output_[PASS_]->flush ();
}


//-----------------------------------------------------------------------
//   getWriter
//-----------------------------------------------------------------------


Writer& PatternLogger::getWriter ( const String& tag )
{
  if ( tag == prevTag_ )
  {
    return *prevOutput_;
  }

  byte   action;
  idx_t  i, n;


  action = BLOCK_;
  n      = lastBlocker_;

  for ( i = 0; i <= n; i++ )
  {
    if ( Pattern::matches( patterns_[i], tag ) )
    {
      action = actions_[i];
    }
  }

  n = patterns_.size ();

  for ( ; i < n && action == BLOCK_; i++ )
  {
    if ( Pattern::matches( patterns_[i], tag ) )
    {
      action = PASS_;
      break;
    }
  }

  prevTag_    = tag;
  prevOutput_ = output_[action].get ();

  return *prevOutput_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void PatternLogger::init_ ( const Array<String>& patterns )
{
  const idx_t  n = patterns.size ();

  String       p;
  idx_t        i, j, k;


  patterns_.resize ( n );
  actions_ .resize ( n );

  lastBlocker_ = -1;

  for ( i = j = 0; i < n; i++ )
  {
    p = patterns[i].stripWhite ();

    if ( p.size() == 0 )
    {
      continue;
    }

    if ( p[0] != '!' )
    {
      patterns_[j] = p;
      actions_ [j] = PASS_;
      j++;
    }
    else
    {
      for ( k = 1; k < p.size() && std::isspace( p[k] ); k++ ) ;

      if ( k < p.size() )
      {
        patterns_[j] = p[slice(k,END)];
        actions_ [j] = BLOCK_;
        lastBlocker_ = j;
        j++;
      }
    }
  }

  if ( j < patterns_.size() )
  {
    patterns_.reshape ( j );
    actions_ .reshape ( j );
  }

  for ( i = 0; i < patterns_.size(); i++ )
  {
    Pattern::check ( patterns_[i] );
  }

  output_[BLOCK_] = newInstance<NullWriter> ();
  prevOutput_     = output_[BLOCK_].get ();
}


JEM_END_PACKAGE( io )
