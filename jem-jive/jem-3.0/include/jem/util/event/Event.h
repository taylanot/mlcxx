
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

#ifndef JEM_UTIL_EVENT_EVENT_H
#define JEM_UTIL_EVENT_EVENT_H

#include <jem/util/event/forward.h>
#include <jem/util/event/EventArguments.h>
#include <jem/util/event/DelegateList.h>
#include <jem/util/event/DelegateIterator.h>
#include <jem/util/event/Delegate.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Event
//-----------------------------------------------------------------------


template <class A, class B, class C>

  class Event

{
 public:

  typedef EventArguments<A,B,C>   Arguments;
  typedef Delegate<Arguments>     DelegateType;


  static const int      ARG_COUNT = Arguments::ARG_COUNT;


  inline                Event         ()       noexcept;
  inline               ~Event         ();

  inline void           emit          () const;

  inline void           emit

    ( A                   a )            const;

  inline void           emit

    ( A                   a,
      B                   b )            const;

  inline void           emit

    ( A                   a,
      B                   b,
      C                   c )            const;

  inline void           swap

    ( Event&              rhs )                noexcept;

  inline bool           enable

    ( bool                flag = true );

  inline void           clear         ();
  inline bool           isConnected   () const noexcept;

  void                  connectTo

    ( DelegateType*       d );

  void                  disconnectFrom

    ( DelegateType*       d );


 private:

                        Event

    ( const Event&        rhs );

  Event&                operator =

    ( const Event&        rhs );

  void                  emit_

    ( const Arguments&    args )         const;


 private:

  DelegateList*         dlist_;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class A, class B, class C>

  inline Event<A,B,C>::Event () noexcept

{
  dlist_ = nullptr;
}


template <class A, class B, class C>

  inline Event<A,B,C>::~Event ()

{
  DelegateList::delRef ( dlist_ );
  dlist_ = nullptr;
}


//-----------------------------------------------------------------------
//   emit
//-----------------------------------------------------------------------


template <class A, class B, class C>

  inline void Event<A,B,C>::emit () const

{
  if ( dlist_ )
  {
    emit_ ( Arguments() );
  }
}


template <class A, class B, class C>

  inline void Event<A,B,C>::emit ( A a ) const

{
  if ( dlist_ )
  {
    emit_ ( Arguments( a ) );
  }
}


template <class A, class B, class C>

  inline void Event<A,B,C>::emit ( A a, B b ) const

{
  if ( dlist_ )
  {
    emit_ ( Arguments( a, b ) );
  }
}


template <class A, class B, class C>

  inline void Event<A,B,C>::emit ( A a, B b, C c ) const

{
  if ( dlist_ )
  {
    emit_ ( Arguments( a, b, c ) );
  }
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class A, class B, class C>

  inline void Event<A,B,C>::swap ( Event& rhs ) noexcept

{
  DelegateList*  tmp = dlist_;

  dlist_     = rhs.dlist_;
  rhs.dlist_ = tmp;
}


//-----------------------------------------------------------------------
//   enable
//-----------------------------------------------------------------------


template <class A, class B, class C>

  inline bool Event<A,B,C>::enable ( bool flag )

{
  if ( dlist_ )
  {
    return dlist_->enable ( flag );
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class A, class B, class C>

  inline void Event<A,B,C>::clear ()

{
  if ( dlist_ )
  {
    dlist_->clear ();
  }
}


//-----------------------------------------------------------------------
//   isConnected
//-----------------------------------------------------------------------


template <class A, class B, class C>

  inline bool Event<A,B,C>::isConnected () const noexcept

{
  if ( dlist_ )
  {
    return ! dlist_->isEmpty ();
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   connectTo
//-----------------------------------------------------------------------


template <class A, class B, class C>

  void Event<A,B,C>::connectTo ( DelegateType* d )

{
  if ( ! dlist_ )
  {
    dlist_ = new DelegateList ();
  }

  dlist_->pushBack ( d );
}


//-----------------------------------------------------------------------
//   disconnectFrom
//-----------------------------------------------------------------------


template <class A, class B, class C>

  void Event<A,B,C>::disconnectFrom ( DelegateType* d )

{
  if ( dlist_ )
  {
    dlist_->erase ( d );
  }
  else
  {
    AbstractDelegate::free ( d );
  }
}


//-----------------------------------------------------------------------
//   emit_
//-----------------------------------------------------------------------


template <class A, class B, class C>

  void Event<A,B,C>::emit_ ( const Arguments& args ) const

{
  DelegateIterator  it ( dlist_ );
  DelegateType*     d;

  d = static_cast<DelegateType*> ( it.get() );

  while ( d )
  {
    d->call ( args );
    d = static_cast<DelegateType*> ( it.next() );
  }
}


JEM_END_PACKAGE( util )

#endif
