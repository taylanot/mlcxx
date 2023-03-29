
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

#ifndef JEM_UTIL_SINGLETON_H
#define JEM_UTIL_SINGLETON_H

#include <jem/base/utilities.h>
#include <jem/base/Once.h>
#include <jem/base/Object.h>
#include <jem/base/SpinLock.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Singleton
//-----------------------------------------------------------------------


template <class T, class D = T>

  class Singleton

{
 public:

  static inline Ref<T>  get       ();

  static inline void    set

    ( const Ref<T>&       obj );


 private:

  class                 Data_
  {
   public:

    SpinLock              lock;
    Ref<T>                instance;
  };


  static void           init_     ();
  static Data_&         getData_  ();


  static Data_*         data_;

};





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Singleton
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


template <class T, class D>

  typename Singleton<T,D>::Data_* Singleton<T,D>::data_ = nullptr;


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


template <class T, class D>

  inline Ref<T> Singleton<T,D>::get ()

{
  Data_&  data ( getData_() );
  Ref<T>  obj;

  {
    Lock<SpinLock> lock ( data.lock );

    obj = data.instance;
  }

  return obj;
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


template <class T, class D>

  inline void Singleton<T,D>::set ( const Ref<T>& obj )

{
  Data_&          data ( getData_() );
  Lock<SpinLock>  lock ( data.lock  );

  data.instance = obj;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T, class D>

  void Singleton<T,D>::init_ ()

{
  data_ = new Data_ ();

  data_->instance = newInstance<D> ();
}


//-----------------------------------------------------------------------
//   getData_
//-----------------------------------------------------------------------


template <class T, class D>

  typename Singleton<T,D>::Data_& Singleton<T,D>::getData_ ()

{
  static Once once = JEM_ONCE_INITIALIZER;

  runOnce ( once, & init_ );

  return * data_;
}


JEM_END_PACKAGE( util )

#endif
