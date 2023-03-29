
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


#include <utility>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/RuntimeException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Flex.h>
#include <jem/util/HashMap.h>
#include <jem/util/FastAllocator.h>
#include <jem/util/HashDictionary.h>
#include <jem/xutil/SnoopDictionary.h>


JEM_DEFINE_SERIABLE_CLASS( jem::xutil::SnoopDictionary );


JEM_BEGIN_PACKAGE( xutil )


using jem::io::ObjectInput;
using jem::io::ObjectOutput;
using jem::util::makePair;
using jem::util::Flex;
using jem::util::HashMap;
using jem::util::HashDict;
using jem::util::FastAllocator;

typedef Ref<Object>       ObjRef;
typedef Flex<SnoopDict*>  SnoopStack;


//=======================================================================
//   class SnoopDictionary::Entry_
//=======================================================================


class SnoopDictionary::Entry_
{
 public:

  inline            Entry_  ();

  inline            Entry_

    ( const Entry_&   rhs );

  inline            Entry_

    ( Entry_&&        rhs );


 public:

  Ref<SnoopDict>    snoop;

  int               readCount;
  int               writeCount;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline SnoopDictionary::Entry_::Entry_ ()
{
  readCount = writeCount = 0;
}


inline SnoopDictionary::Entry_::Entry_ ( const Entry_& rhs ) :

  snoop      ( rhs.snoop      ),
  readCount  ( rhs.readCount  ),
  writeCount ( rhs.writeCount )

{}


inline SnoopDictionary::Entry_::Entry_ ( Entry_&& rhs ) :

  snoop      ( std::move( rhs.snoop ) ),
  readCount  ( rhs.readCount ),
  writeCount ( rhs.writeCount )

{}


//=======================================================================
//   class SnoopDictionary::Map_
//=======================================================================


class SnoopDictionary::Map_ :

  public HashMap < String, Entry_, FastAllocator >

{};


//=======================================================================
//   class SnoopDictionary
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  SnoopDictionary::MAX_DEPTH = 256;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SnoopDictionary::SnoopDictionary ()
{
  map_ = nullptr;
}


SnoopDictionary::SnoopDictionary ( const Ref<Dict>& dict ) :

  dict_ ( dict )

{
  JEM_PRECHECK ( dict );

  map_ = nullptr;
}


SnoopDictionary::~SnoopDictionary ()
{
  delete map_;
  map_ = nullptr;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void SnoopDictionary::readFrom ( ObjectInput& in )
{
  uidx_t  sz;

  dict_ = checkedCast<Dict> ( in.decodeObject() );

  decode ( in, sz );

  delete map_;

  map_ = 0;

  if ( sz )
  {
    const idx_t  n = (idx_t) sz;

    map_ = new Map_ ();

    map_->reserve ( n );

    for ( idx_t i = 0; i < n; i++ )
    {
      String  key;
      Entry_  e;

      decode ( in, key, e.readCount, e.writeCount, e.snoop );

      map_->insert ( std::move( key ), std::move( e ) );
    }

    map_->trimToSize ();
  }
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void SnoopDictionary::writeTo ( ObjectOutput& out ) const
{
  out.encodeObject ( dict_ );

  if ( map_ )
  {
    Map_::Iterator  it  = map_->begin ();
    Map_::Iterator  end = map_->end   ();

    encode ( out, map_->size() );

    for ( ; it != end; ++it )
    {
      Entry_&  e = it->second;

      encode ( out, it->first, e.readCount, e.writeCount, e.snoop );
    }
  }
  else
  {
    encode ( out, 0_idx );
  }
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> SnoopDictionary::clone () const
{
  if ( ! dict_ )
  {
    return newInstance<Self> ();
  }
  else
  {
    return newInstance<Self> ( checkedCast<Dict>( dict_->clone() ) );
  }
}


//-----------------------------------------------------------------------
//   getDictEnum
//-----------------------------------------------------------------------


Ref<DictEnum> SnoopDictionary::getDictEnum () const
{
  if ( ! dict_ )
  {
    initDict_ ();
  }

  return dict_->getDictEnum ();
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool SnoopDictionary::contains ( const String& key ) const
{
  if ( ! dict_ )
  {
    return false;
  }
  else
  {
    return dict_->contains ( key );
  }
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


bool SnoopDictionary::find

  ( ObjRef&        value,
    const String&  key ) const

{
  if ( dict_ == nullptr || ! dict_->find( value, key ) )
  {
    return false;
  }

  if ( ! map_ )
  {
    const_cast<Self*>(this)->map_ = new Map_ ();
  }

  Entry_&  e = (*map_)[key];

  e.readCount++;

  if ( e.snoop )
  {
    if ( e.snoop == value )
    {
      return true;
    }

    if ( e.snoop->dict_ == value )
    {
      value = e.snoop;

      return true;
    }

    e.snoop = nullptr;
  }

  if ( ! value )
  {
    return true;
  }

  if ( dynamicCast<Dict*>( value ) )
  {
    if ( dynamicCast<Self*>( value ) )
    {
      e.snoop = staticCast<SnoopDict> ( value );
    }
    else
    {
      e.snoop = newInstance<SnoopDict> ( staticCast<Dict>( value ) );
      value   = e.snoop;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void SnoopDictionary::insert

  ( const String&  key,
    const ObjRef&  value )

{
  insert ( String( key ), ObjRef( value ) );
}


void SnoopDictionary::insert

  ( String&&  key,
    ObjRef&&  value )

{
  if ( ! dict_ )
  {
    initDict_ ();
  }

  if ( ! map_ )
  {
    map_ = new Map_ ();
  }

  Entry_&  e = (*map_)[key];

  e.writeCount++;

  if ( dynamicCast<Self*>( value ) )
  {
    Self*  self = staticCast<Self*> ( value );

    if ( ! self->dict_ )
    {
      self->initDict_ ();
    }

    e.snoop = self;

    dict_->insert ( std::move( key ), self->dict_ );

    return;
  }

  if ( e.snoop != nullptr && e.snoop->dict_ != value )
  {
    e.snoop = nullptr;
  }

  dict_->insert ( std::move( key ), std::move( value ) );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void SnoopDictionary::erase ( const String& key )
{
  if ( dict_ )
  {
    dict_->erase ( key );
  }

  if ( map_ )
  {
    map_ ->erase ( key );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SnoopDictionary::clear ()
{
  if ( dict_ )
  {
    dict_->clear ();
  }

  if ( map_ )
  {
    map_ ->clear ();
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void SnoopDictionary::reserve ( idx_t count )
{
  if ( dict_ )
  {
    dict_->reserve ( count );
  }

  if ( map_ )
  {
    map_ ->reserve ( count );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void SnoopDictionary::trimToSize ()
{
  if ( dict_ )
  {
    dict_->trimToSize ();
  }

  if ( map_ )
  {
    map_ ->trimToSize ();
  }
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


idx_t SnoopDictionary::capacity () const
{
  if ( ! dict_ )
  {
    return Super::capacity ();
  }
  else
  {
    return dict_->capacity ();
  }
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t SnoopDictionary::size () const
{
  if ( ! dict_ )
  {
    return 0;
  }
  else
  {
    return dict_->size ();
  }
}


//-----------------------------------------------------------------------
//   sync
//-----------------------------------------------------------------------


void SnoopDictionary::sync ()
{
  if ( ! dict_ )
  {
    if ( map_ )
    {
      delete map_;
      map_ = nullptr;
    }

    return;
  }

  if ( ! map_ )
  {
    return;
  }

  SnoopDict*  snoop = this;
  SnoopStack  stack;

  while ( snoop )
  {
    Map_::Iterator  it  = snoop->map_->begin ();
    Map_::Iterator  end = snoop->map_->end   ();

    while ( it != end )
    {
      if ( ! snoop->dict_->contains( it->first ) )
      {
        snoop->map_->erase ( it++ );
      }
      else
      {
        if ( it->second.snoop )
        {
          stack.pushBack ( it->second.snoop.get() );

          if ( stack.size() > MAX_DEPTH )
          {
            depthError_ ( JEM_FUNC );
          }
        }

        ++it;
      }
    }

    if ( stack.size() == 0 )
    {
      snoop = 0;
    }
    else
    {
      snoop = stack.back ();
      stack.popBack ();
    }
  }
}


//-----------------------------------------------------------------------
//   resetCounts
//-----------------------------------------------------------------------


void SnoopDictionary::resetCounts ()
{
  StdFilter  filter;

  resetCounts ( filter );
}


//-----------------------------------------------------------------------
//   resetCounts
//-----------------------------------------------------------------------


void SnoopDictionary::resetCounts ( const Filter& filter )
{
  if ( ! map_ )
  {
    return;
  }

  SnoopDict*  snoop = this;
  SnoopStack  stack;

  while ( snoop )
  {
    Map_::Iterator  it  = snoop->map_->begin ();
    Map_::Iterator  end = snoop->map_->end   ();

    for ( ; it != end; ++it )
    {
      Entry_&  e = it->second;

      if ( e.readCount  > 0 && filter.accept( e.readCount, 0 ) )
      {
        e.readCount = 0;
      }

      if ( e.writeCount > 0 && filter.accept( 0, e.writeCount ) )
      {
        e.writeCount = 0;
      }

      if ( e.snoop )
      {
        stack.pushBack ( e.snoop.get() );

        if ( stack.size() > MAX_DEPTH )
        {
          depthError_ ( JEM_FUNC );
        }
      }
    }

    if ( stack.size() == 0 )
    {
      snoop = 0;
    }
    else
    {
      snoop = stack.back ();
      stack.popBack ();
    }
  }
}


//-----------------------------------------------------------------------
//   resetCounts
//-----------------------------------------------------------------------


void SnoopDictionary::resetCounts ( const String& key )
{
  if ( map_ )
  {
    Map_::Iterator  it = map_->find ( key );

    if ( it != map_->end() )
    {
      Entry_& e = it->second;

      e.readCount = e.writeCount = 0;
    }
  }
}


//-----------------------------------------------------------------------
//   resetReadCount
//-----------------------------------------------------------------------


void SnoopDictionary::resetReadCount ( const String& key )
{
  if ( map_ )
  {
    Map_::Iterator  it = map_->find ( key );

    if ( it != map_->end() )
    {
      it->second.readCount = 0;
    }
  }
}


//-----------------------------------------------------------------------
//   resetWriteCount
//-----------------------------------------------------------------------


void SnoopDictionary::resetWriteCount ( const String& key )
{
  if ( map_ )
  {
    Map_::Iterator  it = map_->find ( key );

    if ( it != map_->end() )
    {
      it->second.writeCount = 0;
    }
  }
}


//-----------------------------------------------------------------------
//   getReadCount
//-----------------------------------------------------------------------


int SnoopDictionary::getReadCount ( const String& key ) const
{
  if ( map_ )
  {
    Map_::Iterator  it = map_->find ( key );

    if ( it != map_->end() )
    {
      return it->second.readCount;
    }
  }

  return 0;
}


//-----------------------------------------------------------------------
//   getWriteCount
//-----------------------------------------------------------------------


int SnoopDictionary::getWriteCount ( const String& key ) const
{
  if ( map_ )
  {
    Map_::Iterator  it = map_->find ( key );

    if ( it != map_->end() )
    {
      return it->second.writeCount;
    }
  }

  return 0;
}


//-----------------------------------------------------------------------
//   listObjects
//-----------------------------------------------------------------------


Array<String> SnoopDictionary::listObjects

  ( const Filter&  filter,
    const String&  sep ) const

{
  SnoopDict*     snoop = const_cast<Self*> ( this );
  Dict*          dict  = dict_.get ();

  Flex<String>   list;
  Flex<String>   nameStack;
  Flex<Dict*>    dictStack;

  String         path;


  while ( dict )
  {
    Ref<DictEnum>  denum = dict->enumerate ();

    for ( ; ! denum->atEnd(); denum->toNext() )
    {
      String  name       = denum->getKey   ();
      ObjRef  value      = denum->getValue ();

      Dict*   nextDict   = 0;
      int     readCount  = 0;
      int     writeCount = 0;

      if ( snoop && snoop->map_ )
      {
        Map_::Iterator  it = snoop->map_->find ( name );

        if ( it != snoop->map_->end() )
        {
          Entry_&  e = it->second;

          readCount  = e.readCount;
          writeCount = e.writeCount;

          if ( e.snoop )
          {
            if ( e.snoop != value && e.snoop->dict_ != value )
            {
              e.snoop = nullptr;
            }
            else
            {
              nextDict = e.snoop.get ();
            }
          }
        }
      }

      if ( filter.accept( readCount, writeCount ) )
      {
        list.pushBack ( join_( path, sep, name ) );
      }

      if ( ! nextDict )
      {
        nextDict = dynamicCast<Dict*> ( value );
      }

      if ( nextDict )
      {
        nameStack.pushBack ( join_( path, sep, name ) );
        dictStack.pushBack ( nextDict );

        if ( dictStack.size() > MAX_DEPTH )
        {
          depthError_ ( JEM_FUNC );
        }
      }
    }

    snoop = 0;

    if ( ! dictStack.size() )
    {
      dict = 0;
    }
    else
    {
      dict  = dictStack.back ();
      path  = nameStack.back ();
      snoop = ClassTemplate<Self>::dynamicCast ( *dict );

      dictStack.popBack ();
      nameStack.popBack ();
    }
  }

  return Array<String> ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   listTouchedObjects
//-----------------------------------------------------------------------


Array<String> SnoopDictionary::listTouchedObjects

  ( const String& sep ) const

{
  StdFilter  filter;

  filter.minReadCount  = 1;
  filter.minWriteCount = 1;

  return listObjects ( filter, sep );
}


//-----------------------------------------------------------------------
//   listUntouchedObjects
//-----------------------------------------------------------------------


Array<String> SnoopDictionary::listUntouchedObjects

  ( const String& sep ) const

{
  StdFilter  filter ( StdFilter::AND_MODE );

  filter.maxReadCount  = 0;
  filter.maxWriteCount = 0;

  return listObjects ( filter, sep );
}


//-----------------------------------------------------------------------
//   initDict_
//-----------------------------------------------------------------------


void SnoopDictionary::initDict_ () const
{
  const_cast<Self*>(this)->dict_ = newInstance<HashDict> ();
}


//-----------------------------------------------------------------------
//   join_
//-----------------------------------------------------------------------


String SnoopDictionary::join_

  ( const String&  path,
    const String&  sep,
    const String&  name )

{
  const idx_t  m = path.size ();
  const idx_t  n = sep .size ();
  const idx_t  p = name.size ();

  if ( m == 0 )
  {
    return name;
  }

  if ( p == 0 )
  {
    return path;
  }

  StringBuffer  buf;

  buf.reserve ( m + n + p );

  buf += path;
  buf += sep;
  buf += name;

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   depthError_
//-----------------------------------------------------------------------


void SnoopDictionary::depthError_ ( const char* where ) const
{
  throw RuntimeException (
    where,
    String::format (
      "maximum depth (%d) exceeded (circular reference?)",
      MAX_DEPTH
    )
  );
}


//=======================================================================
//   class SnoopDictionary::Filter
//=======================================================================


SnoopDictionary::Filter::~Filter ()
{}


//=======================================================================
//   class SnoopDictionary::StdFilter
//=======================================================================


SnoopDictionary::StdFilter::StdFilter ( Mode mode )
{
  this->mode    = mode;

  minReadCount  = 0;
  maxReadCount  = maxOf ( maxReadCount );
  minWriteCount = 0;
  maxWriteCount = maxOf ( maxWriteCount );
}


//-----------------------------------------------------------------------
//   accept
//-----------------------------------------------------------------------


bool SnoopDictionary::StdFilter::accept

  ( int  rcount,
    int  wcount ) const

{
  bool  readTest  = (rcount  >= minReadCount &&
                     rcount  <= maxReadCount);

  bool  writeTest = (wcount >= minWriteCount &&
                     wcount <= maxWriteCount);

  bool  result;


  switch ( mode )
  {
  case OR_MODE:

    result = (readTest || writeTest);

    break;

  case XOR_MODE:

    if ( readTest && writeTest )
    {
      result = false;
    }
    else
    {
      result = (readTest || writeTest);
    }

    break;

  case AND_MODE:

    result = (readTest && writeTest);

    break;

  default:

    result = false;
  }

  return result;
}


JEM_END_PACKAGE( xutil )
