
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

#ifndef JEM_UTIL_EVENT_METHODDELEGATE_H
#define JEM_UTIL_EVENT_METHODDELEGATE_H

#include <jem/base/WeakRef.h>
#include <jem/util/event/DelegateTraits.h>
#include <jem/util/event/Delegate.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class MethodDelegate
//-----------------------------------------------------------------------


template <class Args, class T, class M>

  class MethodDelegate : public Delegate<Args>

{
 public:

  typedef M             Method;


  inline                MethodDelegate

    ( const WeakRef<T>&   obj,
      Method              method );

  virtual bool          isConnected    () const override;

  virtual bool          equals

    ( AbstractDelegate*   rhs  )          const override;

  virtual void          call

    ( const Args&         args )                override;


 protected:

  virtual              ~MethodDelegate ();


 private:

  WeakRef<T>            object_;
  Method                method_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class E, class T, class M>

  inline void           connect

  ( E&                    event,
    T*                    obj,
    M                     method );


template <class E, class T, class M>

  inline void            connect

  ( E&                     event,
    const Ref<T>&          obj,
    M                      method );


template <class E, class T, class M>

  inline void           disconnect

  ( E&                    event,
    T*                    obj,
    M                     method );


template <class E, class T, class M>

  inline void            disconnect

  ( E&                     event,
    const Ref<T>&          obj,
    M                      method );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class MethodDelegate
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class A, class T, class M>

  inline MethodDelegate<A,T,M>::MethodDelegate

  ( const WeakRef<T>&  obj,
    Method             method ) :

    object_ ( obj    ),
    method_ ( method )

{
  if ( method_ == 0 )
  {
    object_ = Ref<T> ();
  }
}


template <class A, class T, class M>

  MethodDelegate<A,T,M>::~MethodDelegate ()

{}


//-----------------------------------------------------------------------
//   isConnected
//-----------------------------------------------------------------------


template <class A, class T, class M>

  bool MethodDelegate<A,T,M>::isConnected () const

{
  return (object_.get() != nullptr);
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


template <class A, class T, class M>

  bool MethodDelegate<A,T,M>::equals ( AbstractDelegate* rhs ) const

{
  MethodDelegate*  md = dynamic_cast<MethodDelegate*> ( rhs );

  if ( md )
  {
    return ( method_       == md->method_ &&
             object_.get() == md->object_.get() );
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   call
//-----------------------------------------------------------------------


template <class A, class T, class M>

  void MethodDelegate<A,T,M>::call ( const A& args )

{
  Ref<T>  tmp = object_.get ();

  if ( tmp )
  {
    DelegateTraits<M>::call ( method_, * tmp, args );
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   connect
//-----------------------------------------------------------------------


template <class E, class T, class M>

  inline void connect ( E& event, T* obj, M method )

{
  event.connectTo (
    new MethodDelegate<typename E::Arguments,T,M> (
      Ref<T> ( obj ),
      method
    )
  );
}


template <class E, class T, class M>

  inline void connect ( E& event, const Ref<T>& obj, M method )

{
  event.connectTo (
    new MethodDelegate<typename E::Arguments,T,M> (
      obj,
      method
    )
  );
}


//-----------------------------------------------------------------------
//   disconnect
//-----------------------------------------------------------------------


template <class E, class T, class M>

  inline void disconnect ( E& event, T* obj, M method )

{
  event.disconnectFrom (
    new MethodDelegate<typename E::Arguments,T,M> (
      Ref<T> ( obj ),
      method
    )
  );
}


template <class E, class T, class M>

  inline void disconnect ( E& event, const Ref<T>& obj, M method )

{
  event.disconnectFrom (
    new MethodDelegate<typename E::Arguments,T,M> (
      obj,
      method
    )
  );
}


JEM_END_PACKAGE( util )

#endif
