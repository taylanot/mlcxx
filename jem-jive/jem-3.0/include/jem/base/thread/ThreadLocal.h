
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

#ifndef JEM_BASE_THREAD_THREADLOCAL_H
#define JEM_BASE_THREAD_THREADLOCAL_H

#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ThreadLocal
//-----------------------------------------------------------------------


class ThreadLocal : public Object
{
 public:

  typedef ThreadLocal   Self;
  typedef Object        Super;


  static const int      ROW_COUNT    =  8;
  static const int      COL_COUNT    = 16;
  static const int      MAX_KEYS     = ROW_COUNT * COL_COUNT;

  static const int      BIT_MASK     = COL_COUNT - 1;
  static const int      SHIFT_BITS   =  4;
  static const int      MAX_ITER     = 64;


                        ThreadLocal   ();

  Ref<ThreadLocal>      clone         () const;

  inline Ref<Object>    get

    ( int                 key )          const;

  void                  set

    ( int                 key,
      const Ref<Object>&  obj );

  inline Object*        getLocale     () const;
  inline Object*        getSysdata    () const;

  void                  setLocale

    ( const Ref<Object>&  obj );

  void                  setSysdata

    ( const Ref<Object>&  obj );


 protected:

  virtual              ~ThreadLocal   ();


 private:

  Ref<Object>*          data_[ROW_COUNT];
  Ref<Object>           locale_;
  Ref<Object>           sysdata_;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline Ref<Object> ThreadLocal::get ( int key ) const
{
  const int  i = key >> SHIFT_BITS;
  const int  j = key &  BIT_MASK;

  if ( data_[i] )
  {
    return data_[i][j];
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   getLocale
//-----------------------------------------------------------------------


inline Object* ThreadLocal::getLocale () const
{
  return locale_.get ();
}


//-----------------------------------------------------------------------
//   getSysdata
//-----------------------------------------------------------------------


inline Object* ThreadLocal::getSysdata () const
{
  return sysdata_.get ();
}


JEM_END_PACKAGE_BASE

#endif
