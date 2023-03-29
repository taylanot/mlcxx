
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
#include <jem/util/event/DelegateList.h>
#include <jem/util/event/DelegateIterator.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class DelegateIterator
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DelegateIterator::DelegateIterator ( DelegateList* dlist )
{
  JEM_ASSERT2 ( dlist != nullptr, "NULL DelegateList pointer" );

  dlist_ = dlist;

  if ( dlist->refcount_ == maxOf( dlist->refcount_ ) )
  {
    throw RuntimeException ( JEM_FUNC, "reference count overflow" );
  }

  dlist->refcount_++;

  if ( dlist->iterCount_ == maxOf( dlist->iterCount_ ) )
  {
    throw RuntimeException (
      JEM_FUNC,
      "event invoked recursively too many times"
    );
  }

  dlist->iterCount_++;

  if ( dlist->status_ & DelegateList::IS_ENABLED_ )
  {
    current_ = dlist->first_;
  }
  else
  {
    current_ = nullptr;
  }
}


DelegateIterator::~DelegateIterator ()
{
  DelegateList&   dlist = * dlist_;


  dlist.refcount_--;
  dlist.iterCount_--;

  if ( dlist.iterCount_ == 0 )
  {
    if ( dlist.status_ & DelegateList::CLEAR_PENDING_ )
    {
      dlist.clear_ ( dlist.first_ );

      dlist.status_  = (byte)
        (dlist.status_ & (~DelegateList::CLEAR_PENDING_));
      dlist.gcCount_ = 0;
    }
    else if ( dlist.eraseList_ )
    {
      AbstractDelegate*  d = dlist.eraseList_;

      while ( d )
      {
        dlist.eraseList_ = d->next_;

        dlist.erase_ ( d );

        d = dlist.eraseList_;
      }
    }
    else if ( dlist.gcCount_ > DelegateList::MAX_GC_COUNT_ )
    {
      dlist.erase_ ( nullptr );
    }
  }

  if ( dlist.refcount_ == 0 )
  {
    delete dlist_;
  }

  dlist_ = nullptr;
}


JEM_END_PACKAGE( util )
