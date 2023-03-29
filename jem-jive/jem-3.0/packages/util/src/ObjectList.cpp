
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


#include <jem/base/memory.h>
#include <jem/base/MemCache.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/error.h>
#include <jem/util/ObjectList.h>


JEM_DEFINE_SERIABLE_CLASS( jem::util::ObjectList );


JEM_BEGIN_PACKAGE( util )


typedef Ref<Object>  ObjRef;


//=======================================================================
//   class ObjectList::Enumerator_
//=======================================================================


class ObjectList::Enumerator_ : public SequenceEnumerator
{
 public:

  typedef ObjectList      List;


  explicit inline         Enumerator_

    ( const Ref<List>&      list );

  virtual bool            atEnd           () const override;
  virtual bool            atBegin         () const override;
  virtual void            toNext          ()       override;
  virtual void            toPrevious      ()       override;

  virtual void            move

    ( idx_t                 count )                override;

  virtual ObjRef          get             () const override;

  virtual void            set

    ( const ObjRef&         obj )                  override;

  virtual void            insert

    ( const ObjRef&         obj )                  override;

  virtual void            erase           ()       override;
  inline void             checkOnList     () const;
  inline void             checkWritable   ();


 protected:

  virtual                ~Enumerator_     ();


 private:

  Ref<List>               list_;
  idx_t                   index_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ObjectList::Enumerator_::Enumerator_ ( const Ref<List>& list ) :

  list_ ( list )

{
  index_ = 0;

  list_->enumCount_.fetch_add ( 1, std::memory_order_relaxed );
}


ObjectList::Enumerator_::~Enumerator_ ()
{
  list_->enumCount_.fetch_sub ( 1, std::memory_order_relaxed );
}


//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


bool ObjectList::Enumerator_::atEnd () const
{
  return (index_ >= list_->size_);
}


//-----------------------------------------------------------------------
//   atBegin
//-----------------------------------------------------------------------


bool ObjectList::Enumerator_::atBegin () const
{
  return (index_ == 0);
}


//-----------------------------------------------------------------------
//   toNext
//-----------------------------------------------------------------------


void ObjectList::Enumerator_::toNext ()
{
  checkOnList ();

  index_++;
}


//-----------------------------------------------------------------------
//   toPrevious
//-----------------------------------------------------------------------


void ObjectList::Enumerator_::toPrevious ()
{
  if ( index_ == 0 )
  {
    moveError_ ( -1 );
  }

  index_--;
}


//-----------------------------------------------------------------------
//   move
//-----------------------------------------------------------------------


void ObjectList::Enumerator_::move ( idx_t count )
{
  idx_t  i = index_ + count;

  if ( i < 0 || i > list_->size_ )
  {
    moveError_ ( count );
  }

  index_ = i;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


ObjRef ObjectList::Enumerator_::get () const
{
  checkOnList ();

  idx_t  i = list_->ipos_ ( index_ );

  return list_->items_[i];
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


void ObjectList::Enumerator_::set ( const ObjRef& obj )
{
  checkOnList ();

  idx_t  i = list_->ipos_ ( index_ );

  list_->items_[i] = obj;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void ObjectList::Enumerator_::insert ( const ObjRef& obj )
{
  checkWritable ();

  list_->insert ( index_, obj );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void ObjectList::Enumerator_::erase ()
{
  checkOnList   ();
  checkWritable ();

  list_->erase ( index_ );
}


//-----------------------------------------------------------------------
//   checkOnList
//-----------------------------------------------------------------------


inline void ObjectList::Enumerator_::checkOnList () const
{
  if ( index_ >= list_->size_ )
  {
    atEndError_ ();
  }
}


//-----------------------------------------------------------------------
//   checkWritable
//-----------------------------------------------------------------------


inline void ObjectList::Enumerator_::checkWritable ()
{
  if ( list_->enumCount_.load( std::memory_order_relaxed ) > 0 )
  {
    writeError_ ();
  }
}


//=======================================================================
//   class ObjectList
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ObjectList::ObjectList () :

  enumCount_ ( 0 )

{
  items_    = nullptr;
  head_     = 0;
  size_     = 0;
  capacity_ = 0;
}


ObjectList::~ObjectList ()
{
  if ( items_ )
  {
    destroy           ( items_, capacity_ );
    MemCache::dealloc ( items_, (size_t) capacity_ *
                                sizeof(ObjRef) );
  }

  items_ = nullptr;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ObjectList::readFrom ( io::ObjectInput& in )
{
  Super::readFrom_ ( in );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ObjectList::writeTo ( io::ObjectOutput& out ) const
{
  Super::writeTo_ ( out );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


ObjRef ObjectList::clone () const
{
  Ref<Self>  list = newInstance<Self> ();

  list->clone_ ( *this );

  return list;
}


//-----------------------------------------------------------------------
//   getSeqEnum
//-----------------------------------------------------------------------


Ref<SeqEnum> ObjectList::getSeqEnum () const
{
  Ref<Self>  self = const_cast<Self*> ( this );

  return newInstance<Enumerator_> ( self );
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void ObjectList::insert

  ( idx_t          idx,
    const ObjRef&  obj )

{
  checkWritable_ ();

  if ( idx < 0 || idx > size_ )
  {
    indexError_ ( idx, size_ + 1 );
  }

  if ( size_ >= capacity_ )
  {
    realloc_ ( expand_( capacity_ ) );
  }

  ObjRef*  items = items_;

  idx_t    i, j;

  if ( idx < size_ / 2 )
  {
    // Shift the items to the left.

    i = j = head_;

    decr_ ( j );

    head_ = j;

    for ( idx_t k = 0; k < idx; k++ )
    {
      items[j].swap ( items[i] );

      j = i;

      incr_ ( i );
    }
  }
  else
  {
    // Shift the items to the right.

    i = j = ipos_ ( size_ );

    for ( idx_t k = size_; k > idx; k-- )
    {
      decr_ ( i );

      items[j].swap ( items[i] );

      j = i;
    }
  }

  items[i] = obj;

  size_++;
}


//-----------------------------------------------------------------------
//   insertSpace
//-----------------------------------------------------------------------


void ObjectList::insertSpace

  ( idx_t  idx,
    idx_t  count )

{
  JEM_PRECHECK2 ( count >= 0, "invalid space to be inserted" );

  checkWritable_ ();

  if ( idx < 0 || idx > size_ )
  {
    indexError_ ( idx, size_ + 1 );
  }

  if ( size_ + count > capacity_ )
  {
    realloc_ ( expand_( size_ + count ) );
  }

  ObjRef*  items = items_;

  if ( idx < size_ / 2 )
  {
    idx_t  i = head_;
    idx_t  j = head_ - count;

    if ( j < 0 )
    {
      j += capacity_;
    }

    head_ = j;

    for ( idx_t k = 0; k < idx; k++ )
    {
      items[j].swap ( items[i] );

      incr_ ( i );
      incr_ ( j );
    }
  }
  else
  {
    idx_t  i = ipos_ ( size_ );
    idx_t  j = ipos_ ( size_ + count );

    for ( idx_t k = size_; k > idx; k-- )
    {
      decr_ ( i );
      decr_ ( j );

      items[j].swap ( items[i] );
    }
  }

  size_ += count;
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void ObjectList::erase ( idx_t idx )
{
  checkWritable_ ();

  if ( idx < 0 || idx >= size_ )
  {
    indexError_ ( idx );
  }

  ObjRef*  items = items_;

  idx_t    i, j;

  i = j = ipos_ ( idx );

  if ( idx < size_ / 2 )
  {
    // Shift the items to the right.

    for ( idx_t k = 0; k < idx; k++ )
    {
      j = i;

      decr_ ( i );

      items[j].swap ( items[i] );
    }

    head_ = j;
  }
  else
  {
    for ( idx_t k = idx; k < size_; k++ )
    {
      j = i;

      incr_ ( i );

      items[j].swap ( items[i] );
    }
  }

  items[i] = nullptr;

  size_--;
}


//-----------------------------------------------------------------------
//   eraseRange
//-----------------------------------------------------------------------


void ObjectList::eraseRange

  ( idx_t  first,
    idx_t  last )

{
  checkWritable_ ();

  if ( first < 0 || first > size_ )
  {
    indexError_ ( first, size_ + 1 );
  }

  if ( last <= first )
  {
    return;
  }

  if ( last > size_ )
  {
    indexError_ ( last, size_ + 1 );
  }

  ObjRef*  items = items_;
  idx_t    count = last - first;

  if ( first < size_ - last )
  {
    idx_t  i = ipos_ ( first );
    idx_t  j = ipos_ ( last );

    for ( idx_t k = 0; k < first; k++ )
    {
      decr_ ( i );
      decr_ ( j );

      items[j].swap ( items[i] );
    }

    count = size_ - count;

    JEM_ASSERT ( count >= 0 );

    while ( size_ > count )
    {
      items[i] = nullptr;

      size_--;

      incr_ ( i );
    }

    head_ = j;
  }
  else
  {
    idx_t  j = ipos_ ( first );
    idx_t  i = ipos_ ( last );

    for ( idx_t k = last; k < size_; k++ )
    {
      items[j].swap ( items[i] );

      incr_ ( i );
      incr_ ( j );
    }

    count = size_ - count;

    JEM_ASSERT ( count >= 0 );

    while ( size_ > count )
    {
      size_--;

      decr_ ( i );

      items[i] = nullptr;
    }
  }
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


ObjRef ObjectList::get ( idx_t idx ) const
{
  if ( idx < 0 || idx >= size_ )
  {
    indexError_ ( idx );
  }

  return items_[ipos_(idx)];
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


void ObjectList::set

  ( idx_t          idx,
    const ObjRef&  obj )

{
  checkWritable_ ();

  if ( idx < 0 || idx >= size_ )
  {
    indexError_ ( idx );
  }

  items_[ipos_(idx)] = obj;
}


//-----------------------------------------------------------------------
//   add
//-----------------------------------------------------------------------


bool ObjectList::add ( const ObjRef& obj )
{
  checkWritable_ ();

  if ( size_ >= capacity_ )
  {
    realloc_ ( expand_( capacity_ ) );
  }

  items_[ipos_(size_)] = obj;

  size_++;

  return true;
}


//-----------------------------------------------------------------------
//   front
//-----------------------------------------------------------------------


ObjRef ObjectList::front () const
{
  if ( size_ == 0 )
  {
    emptyError_ ();
  }

  return items_[head_];
}


//-----------------------------------------------------------------------
//   back
//-----------------------------------------------------------------------


ObjRef ObjectList::back () const
{
  if ( size_ == 0 )
  {
    emptyError_ ();
  }

  return items_[ipos_(size_)];
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void ObjectList::pushBack ( const ObjRef& obj )
{
  checkWritable_ ();

  if ( size_ >= capacity_ )
  {
    realloc_ ( expand_( capacity_ ) );
  }

  items_[ipos_(size_)] = obj;

  size_++;
}


void ObjectList::pushBack ( ObjRef&& obj )
{
  checkWritable_ ();

  if ( size_ >= capacity_ )
  {
    realloc_ ( expand_( capacity_ ) );
  }

  items_[ipos_(size_)] = std::move ( obj );

  size_++;
}


//-----------------------------------------------------------------------
//   pushFront
//-----------------------------------------------------------------------


void ObjectList::pushFront ( const ObjRef& obj )
{
  checkWritable_ ();

  if ( size_ >= capacity_ )
  {
    realloc_ ( expand_( capacity_ ) );
  }

  decr_ ( head_ );

  items_[head_] = obj;

  size_++;
}


void ObjectList::pushFront ( ObjRef&& obj )
{
  checkWritable_ ();

  if ( size_ >= capacity_ )
  {
    realloc_ ( expand_( capacity_ ) );
  }

  decr_ ( head_ );

  items_[head_] = std::move ( obj );

  size_++;
}


//-----------------------------------------------------------------------
//   popBack
//-----------------------------------------------------------------------


void ObjectList::popBack ()
{
  checkWritable_ ();

  if ( size_ == 0 )
  {
    emptyError_ ();
  }

  size_--;

  items_[ipos_(size_)] = nullptr;
}


//-----------------------------------------------------------------------
//   popFront
//-----------------------------------------------------------------------


void ObjectList::popFront ()
{
  checkWritable_ ();

  if ( size_ == 0 )
  {
    emptyError_ ();
  }

  size_--;

  items_[head_] = nullptr;

  incr_ ( head_ );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ObjectList::clear ()
{
  checkWritable_ ();

  ObjRef*  items = items_;

  idx_t  tail = head_ + size_;

  if ( ! isWrapped_() )
  {
    for ( idx_t i = head_; i < tail; i++ )
    {
      items[i] = nullptr;
    }
  }
  else
  {
    tail -= capacity_;

    for ( idx_t i = head_; i < capacity_; i++ )
    {
      items[i] = nullptr;
    }

    for ( idx_t i = 0; i < tail; i++ )
    {
      items[i] = nullptr;
    }
  }

  size_ = 0;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void ObjectList::reserve ( idx_t count )
{
  JEM_PRECHECK2 ( count >= 0, "invalid reserve count" );

  checkWritable_ ();

  if ( count < size_ )
  {
    count = size_;
  }

  if ( count != capacity_ )
  {
    realloc_ ( count );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void ObjectList::trimToSize ()
{
  checkWritable_ ();

  if ( capacity_ > size_ )
  {
    realloc_ ( size_ );
  }
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


idx_t ObjectList::capacity () const
{
  return capacity_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t ObjectList::size () const
{
  return size_;
}


//-----------------------------------------------------------------------
//   ipos_
//-----------------------------------------------------------------------


inline idx_t ObjectList::ipos_ ( idx_t idx ) const
{
  idx += head_;

  if ( idx >= capacity_ )
  {
    idx -= capacity_;
  }

  return idx;
}


//-----------------------------------------------------------------------
//   incr_
//-----------------------------------------------------------------------


inline void ObjectList::incr_ ( idx_t& idx ) const
{
  idx++;

  if ( idx >= capacity_ )
  {
    idx -= capacity_;
  }
}


//-----------------------------------------------------------------------
//   decr_
//-----------------------------------------------------------------------


inline void ObjectList::decr_ ( idx_t& idx ) const
{
  idx--;

  if ( idx < 0 )
  {
    idx += capacity_;
  }
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


void ObjectList::realloc_ ( idx_t newCap )
{
  if ( newCap < 0 )
  {
    overflowError ( JEM_FUNC, newCap );
  }

  if ( newCap < size_ )
  {
    newCap = size_;
  }

  ObjRef*  newItems = (ObjRef*)

    MemCache::alloc ( (size_t) newCap * sizeof(*newItems) );

  try
  {
    construct ( newItems, newCap );
  }
  catch ( ... )
  {
    MemCache::dealloc ( newItems, (size_t) newCap *
                                  sizeof(*newItems) );
    throw;
  }

  ObjRef*  oldItems = items_;

  for ( idx_t i = 0; i < size_; i++ )
  {
    idx_t  j = ipos_ ( i );

    newItems[i].swap ( oldItems[j] );
  }

  if ( oldItems )
  {
    destroy           ( oldItems, capacity_ );
    MemCache::dealloc ( oldItems, (size_t) capacity_ *
                                  sizeof(*oldItems) );
  }

  items_    = newItems;
  head_     = 0;
  capacity_ = newCap;
}


//-----------------------------------------------------------------------
//   isWrapped_
//-----------------------------------------------------------------------


inline bool ObjectList::isWrapped_ () const
{
  return (head_ + size_ > capacity_);
}


//-----------------------------------------------------------------------
//   checkWritable_
//-----------------------------------------------------------------------


inline void ObjectList::checkWritable_ ()
{
  if ( enumCount_.load( std::memory_order_relaxed ) > 0 )
  {
    writeError_ ();
  }
}


JEM_END_PACKAGE( util )
