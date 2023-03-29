
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/RuntimeException.h>
#include <jem/util/event/AbstractDelegate.h>
#include <jem/util/event/DelegateList.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class DelegateList
//=======================================================================

//-----------------------------------------------------------------------
//   static data members
//-----------------------------------------------------------------------


const byte  DelegateList::MAX_GC_COUNT_  =  10;

const byte  DelegateList::IS_ENABLED_    = 0x1;
const byte  DelegateList::CLEAR_PENDING_ = 0x2;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DelegateList::DelegateList ()
{
  first_     = nullptr;
  eraseList_ = nullptr;
  refcount_  = 1;
  iterCount_ = 0;
  gcCount_   = 0;
  status_    = IS_ENABLED_;
}


DelegateList::~DelegateList()
{
  if ( first_ )
  {
    clear_ ( first_ );
  }

  if ( eraseList_ )
  {
    clear_ ( eraseList_ );
  }
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void DelegateList::pushBack ( AbstractDelegate* d )
{
  JEM_PRECHECK2 ( d != nullptr && d->next_ == nullptr,
                  "invalid Event delegate" );

  d->next_ = first_;
  first_   = d;

  if ( gcCount_ <= MAX_GC_COUNT_ )
  {
    gcCount_++;
  }
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void DelegateList::erase ( AbstractDelegate* d )
{
  JEM_PRECHECK2 ( d != nullptr && d->next_ == nullptr,
                  "invalid Event delegate" );

  if ( iterCount_ == 0 )
  {
    erase_ ( d );
  }
  else
  {
    d->next_   = eraseList_;
    eraseList_ = d;
  }
}


//-----------------------------------------------------------------------
//   enable
//-----------------------------------------------------------------------


bool DelegateList::enable ( bool flag )
{
  bool  isEnabled = (status_ & IS_ENABLED_);

  if ( flag )
  {
    status_ = (byte) (status_ | IS_ENABLED_);
  }
  else
  {
    status_ = (byte) (status_ & (~IS_ENABLED_));
  }

  return isEnabled;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void DelegateList::clear ()
{
  if ( iterCount_ > 0 )
  {
    status_ = (byte) (status_ | CLEAR_PENDING_);
  }
  else
  {
    clear_ ( first_ );

    gcCount_ = 0;
  }
}


//-----------------------------------------------------------------------
//   addRef
//-----------------------------------------------------------------------


void DelegateList::addRef ( Self* self )
{
  if ( self )
  {
    if ( self->refcount_ == maxOf( self->refcount_ ) )
    {
      throw RuntimeException ( JEM_FUNC, "reference count overflow" );
    }

    self->refcount_++;
  }
}


//-----------------------------------------------------------------------
//   delRef
//-----------------------------------------------------------------------


void DelegateList::delRef ( Self* self )
{
  if ( self )
  {
    if ( (--self->refcount_) == 0 )
    {
      delete self;
    }
  }
}


//-----------------------------------------------------------------------
//   erase_
//-----------------------------------------------------------------------


void DelegateList::erase_ ( AbstractDelegate* e )
{
  JEM_ASSERT2 ( iterCount_ == 0, "delegate list is in use" );

  AbstractDelegate*  d = first_;
  AbstractDelegate*  p = nullptr;

  AbstractDelegate*  t;

  bool               kill;


  while ( d )
  {
    if ( e && d->equals( e ) )
    {
      AbstractDelegate::free ( e );

      kill = true;
      e    = nullptr;
    }
    else
    {
      kill = ! d->isConnected ();
    }

    if ( kill )
    {
      t = d;
      d = d->next_;

      if ( p )
      {
        p->next_ = d;
      }
      else
      {
        first_   = d;
      }

      AbstractDelegate::free ( t );
    }
    else
    {
      p = d;
      d = d->next_;
    }
  }

  if ( e )
  {
    AbstractDelegate::free ( e );
  }

  gcCount_ = 0;
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


void DelegateList::clear_ ( AbstractDelegate*& list )
{
  AbstractDelegate*  t;

  while ( list )
  {
    t    = list;
    list = list->next_;

    AbstractDelegate::free ( t );
  }
}


JEM_END_PACKAGE( util )
