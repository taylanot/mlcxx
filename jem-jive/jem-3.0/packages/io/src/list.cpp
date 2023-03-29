
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


#include <jem/io/list.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   static data
//=======================================================================


const BeginList  beginList = { 7 };
const BeginItem  beginItem = { "-", 1 };
const EndItem    endItem   = {};
const EndList    endList   = {};


//=======================================================================
//   print functions
//=======================================================================


void                  print

  ( PrintWriter&        pr,
    const BeginList&    list )

{
  pr.pushSettings   ();
  pr.setIndentWidth ( list.indent );
}


void                  print

  ( PrintWriter&        pr,
    const BeginItem&    item )

{
  idx_t  i, j, n;

  i = n = 0;

  for ( j = 0; j < item.size; j++ )
  {
    if ( item.str[j] == '\n' )
    {
      n = jem::max ( n, j - i );
      i = j + 1;
    }
  }

  n = jem::max ( n, j - i );
  j = item.size;

  if ( j > 0 && item.str[j - 1] == '\n' )
  {
    j--;
  }

  pr.printBlankLine  ();
  pr.write           ( item.str, j );
  pr.incrIndentLevel ();

  n = pr.getIndentWidth() - n;

  if ( n >= 2_idx )
  {
    pr.printSpaces ( n );
  }
  else
  {
    pr.printLine ();
  }
}


void                  print

  ( PrintWriter&        pr,
    const EndItem&      end )

{
  pr.decrIndentLevel ();
  pr.printBlankLine  ();
}


void                  print

  ( PrintWriter&        pr,
    const EndList&      end )

{
  pr.popSettings ();
}


JEM_END_PACKAGE( io )
