
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

#ifndef JEM_BASE_REFLECT_METHODTEMPLATE_H
#define JEM_BASE_REFLECT_METHODTEMPLATE_H

#include <jem/base/NullPointerException.h>
#include <jem/base/reflect/Method.h>
#include <jem/base/reflect/MethodTraits.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   makeMethod
//-----------------------------------------------------------------------


template <class M>

  inline Ref<Method>    makeMethod

  ( const String&         name,
    M                     method );







//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class MethodTemplate
//=======================================================================


template <class M>

  class MethodTemplate : public Method

{
 public:

  static const int      ARG_COUNT = MethodTraits<M>::ARG_COUNT;
  static const bool     IS_STATIC = MethodTraits<M>::IS_STATIC;


  inline                MethodTemplate

    ( const String&       name,
      M                   method );

  virtual bool          isStatic        () const override;
  virtual idx_t         argCount        () const override;
  virtual Class*        getReturnType   () const override;

  virtual void          getArgTypes

    ( Class**             types )          const override;

  virtual Ref<Object>   call

    ( const Ref<Object>*  args  )          const override;


 protected:

  virtual              ~MethodTemplate ();


 private:

  M                     method_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class M>

  inline MethodTemplate<M>::MethodTemplate

  ( const String&  name,
    M              method ) :

    Method  ( name   ),
    method_ ( method )


{
  if ( ! method )
  {
    throw NullPointerException ( JEM_FUNC, "NULL method pointer" );
  }
}


template <class M>

  inline MethodTemplate<M>::~MethodTemplate ()

{}


//-----------------------------------------------------------------------
//   isStatic
//-----------------------------------------------------------------------


template <class M>

  bool MethodTemplate<M>::isStatic () const

{
  return IS_STATIC;
}


//-----------------------------------------------------------------------
//   argCount
//-----------------------------------------------------------------------


template <class M>

  idx_t MethodTemplate<M>::argCount () const

{
  return (idx_t) ARG_COUNT;
}


//-----------------------------------------------------------------------
//   getReturnType
//-----------------------------------------------------------------------


template <class M>

  Class* MethodTemplate<M>::getReturnType () const

{
  return MethodTraits<M>::getReturnType ();
}


//-----------------------------------------------------------------------
//   getArgTypes
//-----------------------------------------------------------------------


template <class M>

  void MethodTemplate<M>::getArgTypes ( Class** types ) const

{
  MethodTraits<M>::getArgTypes ( types );
}


//-----------------------------------------------------------------------
//   call
//-----------------------------------------------------------------------


template <class M>

  Ref<Object> MethodTemplate<M>::call

  ( const Ref<Object>* args ) const

{
  return MethodTraits<M>::call ( method_, args );
}


//=======================================================================
//   makeMethod
//=======================================================================


template <class M>

  inline Ref<Method>    makeMethod

  ( const String&         name,
    M                     method )

{
  return newInstance< MethodTemplate<M> >( name, method );
}


JEM_END_PACKAGE_BASE

#endif
