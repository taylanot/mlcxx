
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

#ifndef JEM_BASE_REFLECT_DEFAULTCONSTRUCTOR_H
#define JEM_BASE_REFLECT_DEFAULTCONSTRUCTOR_H

#include <jem/base/reflect/Constructor.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class DefaultConstructor
//-----------------------------------------------------------------------


template <class T>

  class DefaultConstructor : public Constructor

{
 public:

  typedef Constructor   Super;


  inline                DefaultConstructor  ();

  virtual idx_t         argCount            () const override;

  virtual void          getArgTypes

    ( Class**             types )              const override;

  virtual Ref<Object>   call

    ( const Ref<Object>*  args )               const override;


 protected:

  virtual              ~DefaultConstructor  ();

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T>

  inline DefaultConstructor<T>::DefaultConstructor ()

{}


template <class T>

  DefaultConstructor<T>::~DefaultConstructor ()

{}


//-----------------------------------------------------------------------
//   argCount
//-----------------------------------------------------------------------


template <class T>

  idx_t DefaultConstructor<T>::argCount () const

{
  return 0;
}


//-----------------------------------------------------------------------
//   getArgTypes
//-----------------------------------------------------------------------


template <class T>

  void DefaultConstructor<T>::getArgTypes ( Class** ) const

{}


//-----------------------------------------------------------------------
//   call
//-----------------------------------------------------------------------


template <class T>

  Ref<Object> DefaultConstructor<T>::call

  ( const Ref<Object>* ) const

{
  return newInstance<T> ();
}



JEM_END_PACKAGE_BASE

#endif
