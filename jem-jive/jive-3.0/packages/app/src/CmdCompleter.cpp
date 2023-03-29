
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


#include <jem/base/rtti.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Properties.h>
#include <jem/util/StringUtils.h>
#include <jive/app/Module.h>
#include <jive/app/Actions.h>
#include <jive/app/CmdCompleter.h>


JEM_DEFINE_CLASS( jive::app::CmdCompleter );


JIVE_BEGIN_PACKAGE( app )


//=======================================================================
//   class CmdCompleter
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


CmdCompleter::CmdCompleter ()
{
  searchIndex_ = 0;
}


CmdCompleter::~CmdCompleter ()
{}


//-----------------------------------------------------------------------
//   rehash
//-----------------------------------------------------------------------


void CmdCompleter::rehash

  ( Module&            module,
    const Properties&  globdat )

{
  using jem::sort;
  using jem::toValue;
  using jem::isInstance;
  using jem::Array;
  using jem::util::StringUtils;

  Properties            cmds;

  module.takeAction ( Actions::GET_COMMANDS, cmds, globdat );

  StringVector          cmdList ( cmds.listProps() );
  Array<StringVector>   cmdArgs ( cmdList.size()   );

  StringVector          empty;
  Ref<Object>           obj;
  String                str;


  sort ( cmdList );

  for ( idx_t i = 0; i < cmdList.size(); i++ )
  {
    obj = cmds.get ( cmdList[i] );

    if ( toValue( str, obj ) )
    {
      if ( str.size() )
      {
        cmdArgs[i].ref ( StringUtils::split( str ) );
      }
      else
      {
        cmdArgs[i].ref ( empty );
      }
    }
    else if ( isInstance<StringVector>( obj ) )
    {
      cmdArgs[i].ref ( toValue<const StringVector&>( obj ) );
    }
  }

  cmdList_   .swap ( cmdList );
  cmdArgs_   .swap ( cmdArgs );
  searchList_.swap ( empty );

  searchTerm_  = "";
  searchIndex_ = 0;
  prevMatch_   = "";
}


//-----------------------------------------------------------------------
//   newWord
//-----------------------------------------------------------------------


void CmdCompleter::newWord

  ( const String&  word,
    const String&  context )

{
  searchTerm_  = word;
  searchIndex_ = 0;
  prevMatch_   = "";

  if ( context.size() == 0 )
  {
    searchList_.ref ( cmdList_ );
  }
  else
  {
    searchList_.ref ( StringVector() );

    for ( idx_t i = 0; i < cmdList_.size(); i++ )
    {
      if ( cmdList_[i] == context )
      {
        searchList_.ref ( cmdArgs_[i] );
        break;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   nextCompletion
//-----------------------------------------------------------------------


String CmdCompleter::nextCompletion ()
{
  const idx_t  len = searchTerm_.size ();

  while ( searchIndex_ < searchList_.size() )
  {
    const String&  next = searchList_[searchIndex_++];
    const idx_t    n    = next.size ();

    if ( n < len )
    {
      continue;
    }

    if ( len == 0 || next.startsWith( searchTerm_ ) )
    {
      idx_t  i = jem::min ( len + 1, n );

      while ( i < n && next[i] != '.' && next[i] != '/' )
      {
        i++;
      }

      String  match = next[slice(BEGIN,i)];

      if ( match != prevMatch_ )
      {
        prevMatch_ = match;

        return match;
      }
    }
  }

  return String ();
}


JIVE_END_PACKAGE( app )
