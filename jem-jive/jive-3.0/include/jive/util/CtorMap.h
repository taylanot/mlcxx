
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_UTIL_CTORMAP_H
#define JIVE_UTIL_CTORMAP_H

#include <jem/base/Once.h>
#include <jem/base/Ref.h>
#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class CtorMapImp
//-----------------------------------------------------------------------


class CtorMapImp
{
 public:

  typedef CtorMapImp      Self;


  void                    insert

    ( const String&         type,
      void*                 ctor );

  bool                    contains

    ( const String&         type )           const;

  void*                   find

    ( const String&         type )           const;

  StringVector            listKnownTypes  () const;

  static Self*            newInstance     ();


 protected:

  inline                  CtorMapImp      ();
  inline                 ~CtorMapImp      ();


 private:

  class                   Data_;

  Ref<Data_>              data_;

};


//-----------------------------------------------------------------------
//   class CtorMap
//-----------------------------------------------------------------------


template <class T>

  class CtorMap

{
 public:

  typedef T               Ctor;


  static inline void      insert

    ( const String&         type,
      Ctor                  ctor );

  static inline bool      contains

    ( const String&         type );

  static inline Ctor      find

    ( const String&         type );

  static inline
    StringVector          listKnownTypes  ();


 private:

  typedef CtorMapImp      Imp_;

  static inline Imp_&     getImp_         ();
  static void             init_           ();


 private:

  static Imp_*            imp_;

};






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class CtorMap
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


template <class T>

  CtorMapImp* CtorMap<T>::imp_ = 0;


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


template <class T>

  inline void CtorMap<T>::insert

  ( const String&  type,
    Ctor           ctor )

{
  getImp_().insert ( type, (void*) ctor );
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


template <class T>

  inline bool CtorMap<T>::contains ( const String& type )

{
  return getImp_().contains ( type );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


template <class T>

  inline typename CtorMap<T>::Ctor CtorMap<T>::find

  ( const String&  type )

{
  return (Ctor) getImp_().find ( type );
}


//-----------------------------------------------------------------------
//   listKnownTypes
//-----------------------------------------------------------------------


template <class T>

  inline StringVector CtorMap<T>::listKnownTypes ()

{
  return getImp_().listKnownTypes ();
}


//-----------------------------------------------------------------------
//   getImp_
//-----------------------------------------------------------------------


template <class T>

  inline CtorMapImp& CtorMap<T>::getImp_ ()

{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, & init_ );

  return *imp_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T>

  void CtorMap<T>::init_ ()

{
  imp_ = Imp_::newInstance ();
}


JIVE_END_PACKAGE( util )

#endif
