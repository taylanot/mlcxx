
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

#ifndef JEM_IO_LIST_H
#define JEM_IO_LIST_H

#include <jem/io/PrintWriter.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class BeginList
//-----------------------------------------------------------------------


class BeginList
{
 public:

  inline BeginList        operator  ()

    ( int                   indent )   const;

 public:

  const int               indent;

};


extern const BeginList    beginList;


//-----------------------------------------------------------------------
//   class BeginItem
//-----------------------------------------------------------------------


class BeginItem
{
 public:

  inline BeginItem        operator  ()

    ( const String&         str )   const;


 public:

  const char*             str;
  const idx_t             size;

};


extern const BeginItem    beginItem;


//-----------------------------------------------------------------------
//   class EndItem
//-----------------------------------------------------------------------


struct                    EndItem   {};
extern const EndItem      endItem;


//-----------------------------------------------------------------------
//   class EndList
//-----------------------------------------------------------------------


struct                    EndList   {};
extern const EndList      endList;


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const BeginList&        list );

void                      print

  ( PrintWriter&            pr,
    const BeginItem&        item );

void                      print

  ( PrintWriter&            pr,
    const EndItem&          end );

void                      print

  ( PrintWriter&            pr,
    const EndList&          end );






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class BeginList
//=======================================================================


inline BeginList BeginList::operator () ( int n ) const
{
  BeginList  list = { n };

  return list;
}


//=======================================================================
//   class BeginItem
//=======================================================================


inline BeginItem BeginItem::operator () ( const String& s ) const
{
  BeginItem  item = { s.addr(), s.size() };

  return item;
}


JEM_END_PACKAGE( io )

#endif
