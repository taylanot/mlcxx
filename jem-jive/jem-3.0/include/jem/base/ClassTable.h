
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

#ifndef JEM_BASE_CLASSTABLE_H
#define JEM_BASE_CLASSTABLE_H

#include <jem/base/utilities.h>
#include <jem/base/Class.h>
#include <jem/base/ReadWriteMutex.h>
#include <jem/util/HashTable.h>
#include <jem/util/FastAllocator.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ClassTableTraits
//-----------------------------------------------------------------------


class ClassTableTraits
{
 public:

  typedef String                KeyType;
  typedef Class*                ValueType;
  typedef ValueType&            Reference;
  typedef const ValueType&      ConstReference;
  typedef ValueType*            Pointer;
  typedef const ValueType*      ConstPointer;
  typedef util::FastAllocator   Allocator;


  static const KeyType&   getKey

    ( ValueType             value ) noexcept

  {
    return value->name_;
  }

  static bool             equalKeys

    ( const KeyType&        lhs,
      const KeyType&        rhs )

  {
    return ( lhs.equals( rhs ) );
  }

  static lint             hashValue

    ( const KeyType&        key )

  {
    return key.hashValue ();
  }
};


//-----------------------------------------------------------------------
//   class ClassTable
//-----------------------------------------------------------------------


class ClassTable
{
 public:

  typedef ClassTableTraits         Traits;
  typedef util::HashTable<Traits>  Table;
  typedef Table::Iterator          Iterator;
  typedef Table::ConstIterator     ConstIterator;


  inline                  ClassTable  ();

  inline Class*           find

    ( const String&         name )       const;

  inline void             insert

    ( Class*                type );

  inline void             trimToSize  ();


 private:

  ReadWriteMutex          mutex_;
  Table                   table_;

};




//#######################################################################
//   Implemenatation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ClassTable::ClassTable ()
{}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline Class* ClassTable::find ( const String& name ) const
{
  ReadLock lock ( mutex_ );

  ConstIterator  i = table_.find ( name );

  if ( i == table_.end() )
  {
    return nullptr;
  }
  else
  {
    return *i;
  }
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


inline void ClassTable::insert ( Class* type )
{
  WriteLock lock ( mutex_ );

  table_.insert ( type );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


inline void ClassTable::trimToSize ()
{
  table_.trimToSize ();
}


JEM_END_PACKAGE_BASE

#endif
