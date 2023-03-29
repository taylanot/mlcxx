
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


#include <jem/base/ReadWriteMutex.h>
#include <jem/base/array/utilities.h>
#include <jem/util/HashMap.h>
#include <jem/util/FastAllocator.h>
#include <jive/util/CtorMap.h>


using jem::ReadLock;
using jem::WriteLock;
using jem::ReadWriteMutex;
using jem::util::makePair;
using jem::util::HashMap;
using jem::util::FastAllocator;


JIVE_BEGIN_PACKAGE( util )


typedef HashMap < String,
                  void*,
                  FastAllocator>  CtorHashMap;


//=======================================================================
//   class CtorMapImp::Data_
//=======================================================================


class CtorMapImp::Data_ : public jem::Collectable
{
 public:

  CtorHashMap       map;
  ReadWriteMutex    mutex;

};


//=======================================================================
//   class CtorMapImp
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline CtorMapImp::CtorMapImp ()
{
  data_ = jem::newInstance<Data_> ();
}


inline CtorMapImp::~CtorMapImp ()
{}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void CtorMapImp::insert

  ( const String&  type,
    void*          ctor )

{
  Data_&    d = * data_;

  WriteLock lock ( d.mutex );

  d.map.insert   ( makePair( type, ctor ) );
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool CtorMapImp::contains ( const String& type ) const
{
  Data_&    d = * data_;

  ReadLock  lock ( d.mutex );
  bool      result;

  result = d.map.contains ( type );

  return result;
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


void* CtorMapImp::find ( const String& type ) const
{
  Data_&    d = * data_;

  ReadLock  lock ( d.mutex );

  CtorHashMap::Iterator it = d.map.find ( type );

  if ( it == d.map.end() )
  {
    return 0;
  }
  else
  {
    return it->second;
  }
}


//-----------------------------------------------------------------------
//   listKnownTypes
//-----------------------------------------------------------------------


StringVector CtorMapImp::listKnownTypes () const
{
  Data_&        d = * data_;

  StringVector  list;

  {
    ReadLock  lock ( d.mutex );

    CtorHashMap::Iterator  it    = d.map.begin ();
    CtorHashMap::Iterator  ilast = d.map.end   ();

    list.resize ( d.map.size() );

    for ( idx_t i = 0; it != ilast; ++it )
    {
      list[i++] = it->first;
    }
  }

  sort ( list );

  return list;
}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


CtorMapImp* CtorMapImp::newInstance ()
{
  return new Self;
}


JIVE_END_PACKAGE( util )
