
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

#ifndef JEM_UTIL_EVENT_FUNCTIONDELEGATE_H
#define JEM_UTIL_EVENT_FUNCTIONDELEGATE_H

#include <jem/util/event/DelegateTraits.h>
#include <jem/util/event/Delegate.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class FunctionDelegate
//-----------------------------------------------------------------------


template <class Args, class Func>

  class FunctionDelegate : public Delegate<Args>

{
 public:

  typedef Func          Function;


  explicit inline       FunctionDelegate

    ( Function            func );

  virtual bool          isConnected      () const override;

  virtual bool          equals

    ( AbstractDelegate*   rhs  )            const override;

  virtual void          call

    ( const Args&         args )                  override;


 protected:

  virtual              ~FunctionDelegate ();


 private:

  Function              function_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class E, class F>

  inline void           connect

  ( E&                    event,
    F                     func );

template <class E, class F>

  inline void           disconnect

  ( E&                    event,
    F                     func );






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class FunctionDelegate
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class A, class F>

  inline FunctionDelegate<A,F>::FunctionDelegate ( Function f ) :

    function_ ( f )

{}


template <class A, class F>

  inline FunctionDelegate<A,F>::~FunctionDelegate ()

{}


//-----------------------------------------------------------------------
//   isConnected
//-----------------------------------------------------------------------


template <class A, class F>

  bool FunctionDelegate<A,F>::isConnected () const

{
  return (function_ != nullptr);
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


template <class A, class F>

  bool FunctionDelegate<A,F>::equals ( AbstractDelegate* rhs ) const

{
  FunctionDelegate*  fd = dynamic_cast<FunctionDelegate*> ( rhs );

  if ( fd )
  {
    return (function_ == fd->function_);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   call
//-----------------------------------------------------------------------


template <class A, class F>

  void FunctionDelegate<A,F>::call ( const A& args )

{
  if ( function_ )
  {
    DelegateTraits<F>::call ( function_, args );
  }
}



//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   connect
//-----------------------------------------------------------------------


template <class E, class F>

  inline void connect ( E& event, F func )

{
  event.connectTo (
    new FunctionDelegate<typename E::Arguments,F> ( func )
  );
}


//-----------------------------------------------------------------------
//   disconnect
//-----------------------------------------------------------------------


template <class E, class F>

  inline void disconnect ( E& event, F func )

{
  event.disconnectFrom (
    new FunctionDelegate<typename E::Arguments,F> ( func )
  );
}


JEM_END_PACKAGE( util )

#endif
