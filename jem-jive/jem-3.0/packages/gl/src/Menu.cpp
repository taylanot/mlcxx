
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


#include <jem/base/ClassTemplate.h>
#include <jem/gl/Menu.h>


JEM_DEFINE_CLASS( jem::gl::Menu );


JEM_BEGIN_PACKAGE( gl )


using jem::util::List;


//=======================================================================
//   class Menu
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Menu::Menu ()
{}


Menu::Menu ( const Array<String>&  list )
{
  for ( idx_t i = 0; i < list.size(); i++ )
  {
    entries_.pushBack ( list[i] );
  }
}


Menu::Menu ( const Menu& rhs ) :

  entries_ ( rhs.entries_ )

{}


Menu::~Menu ()
{}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Menu::clone () const
{
  return newInstance<Menu> ( *this );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void Menu::clear ()
{
  entries_.clear ();
}


//-----------------------------------------------------------------------
//   addLine
//-----------------------------------------------------------------------


void Menu::addLine ()
{
  entries_.pushBack ( "" );
}


//-----------------------------------------------------------------------
//   addEntry
//-----------------------------------------------------------------------


void Menu::addEntry ( const String& entry )
{
  if ( entry.size() > 0 )
  {
    List<String>::Iterator  first = entries_.begin ();
    List<String>::Iterator  last  = entries_.end   ();

    for ( ; first != last; ++first )
    {
      if ( (*first) == entry )
      {
        return;
      }
    }
  }

  entries_.pushBack ( entry );
}


//-----------------------------------------------------------------------
//   addOption
//-----------------------------------------------------------------------


void Menu::addOption

  ( const String&  first,
    const String&  second,
    bool           state )

{
  if ( state )
  {
    insertEntry ( second, first );
  }
  else
  {
    insertEntry ( first, second );
  }
}


//-----------------------------------------------------------------------
//   insertEntry
//-----------------------------------------------------------------------


void Menu::insertEntry

  ( const String&  entry,
    const String&  next )

{
  List<String>::Iterator  first = entries_.begin ();
  List<String>::Iterator  last  = entries_.end   ();
  List<String>::Iterator  pos   = entries_.end   ();

  for ( ; first != last; ++first )
  {
    if ( (*first) == next )
    {
      pos = first;
    }

    if ( entry.size() > 0 && (*first) == entry )
    {
      return;
    }
  }

  entries_.insert ( pos, entry );
}


//-----------------------------------------------------------------------
//   eraseEntry
//-----------------------------------------------------------------------


void Menu::eraseEntry ( const String& entry )
{
  List<String>::Iterator  first = entries_.begin ();
  List<String>::Iterator  last  = entries_.end   ();

  for ( ; first != last; ++first )
  {
    if ( (*first) == entry )
    {
      entries_.erase ( first );
      return;
    }
  }
}


//-----------------------------------------------------------------------
//   listEntries
//-----------------------------------------------------------------------


Array<String> Menu::listEntries () const
{
  List<String>::ConstIterator  first = entries_.begin ();
  List<String>::ConstIterator  last  = entries_.end   ();

  Array<String>  list ( entries_.size() );

  for ( idx_t i = 0; first != last; ++first, i++ )
  {
    list[i] = *first;
  }

  return list;
}


JEM_END_PACKAGE( gl )
