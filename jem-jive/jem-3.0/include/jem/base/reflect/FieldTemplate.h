
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

#ifndef JEM_BASE_REFLECT_FIELDTEMPLATE_H
#define JEM_BASE_REFLECT_FIELDTEMPLATE_H

#include <jem/base/NullPointerException.h>
#include <jem/base/reflect/Field.h>
#include <jem/base/reflect/FieldTraits.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   makeField
//-----------------------------------------------------------------------


template <class Getter>

  inline Ref<Field>     makeField

  ( const String&         name,
    Getter                getter );


template <class Getter, class Setter>

  inline Ref<Field>     makeField

  ( const String&         name,
    Getter                getter,
    Setter                setter );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class FieldTemplate
//=======================================================================


template <class G>

  class FieldTemplate : public Field

{
 public:

  typedef G             Getter;


  inline                FieldTemplate

    ( const String&       name,
      Getter              getter );

  virtual bool          isStatic        () const override;
  virtual Class*        getFieldType    () const override;

  virtual Ref<Object>   get

    ( const Ref<Object>&  obj )            const override;


 protected:

   virtual             ~FieldTemplate   ();


 private:

   Getter               getter_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class G>

  inline FieldTemplate<G>::FieldTemplate

  ( const String&  name,
    Getter         getter ) :

    Field   ( name   ),
    getter_ ( getter )

{
  if ( ! getter )
  {
    throw NullPointerException ( JEM_FUNC,
                                 "null Field get function" );
  }
}


template <class G>

  FieldTemplate<G>::~FieldTemplate ()

{}


//-----------------------------------------------------------------------
//   isStatic
//-----------------------------------------------------------------------


template <class G>

  bool FieldTemplate<G>::isStatic () const

{
  return FieldGetterTraits<G>::IS_STATIC;
}


//-----------------------------------------------------------------------
//   getFieldType
//-----------------------------------------------------------------------


template <class G>

  Class* FieldTemplate<G>::getFieldType () const

{
  return FieldGetterTraits<G>::getFieldType ();
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


template <class G>

  Ref<Object> FieldTemplate<G>::get ( const Ref<Object>& obj ) const

{
  return FieldGetterTraits<G>::get ( obj, getter_ );
}



//=======================================================================
//   class SettableFieldTemplate
//=======================================================================


template <class G, class S>

  class SettableFieldTemplate : public Field

{
 public:

  typedef G             Getter;
  typedef S             Setter;


  inline                SettableFieldTemplate

    ( const String&       name,
      Getter              getter,
      Setter              setter );

  virtual bool          isStatic      () const override;
  virtual bool          isSettable    () const override;
  virtual Class*        getFieldType  () const override;

  virtual Ref<Object>   get

    ( const Ref<Object>&  obj )          const override;

  virtual void          set

    ( const Ref<Object>&  obj,
      const Ref<Object>&  val )          const override;


 protected:

  virtual              ~SettableFieldTemplate ();


 private:

  Getter                getter_;
  Setter                setter_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class G, class S>

  inline SettableFieldTemplate<G,S>::SettableFieldTemplate

  ( const String&   name,
    Getter          getter,
    Setter          setter ) :

    Field   ( name   ),
    getter_ ( getter ),
    setter_ ( setter )

{
  static_assert ( FieldGetterTraits<G>::IS_STATIC ==
                  FieldSetterTraits<S>::IS_STATIC,
                  "inconsistent static field get/set functions" );

  if ( ! getter )
  {
    throw NullPointerException ( JEM_FUNC,
                                 "NULL Field get function" );
  }
  if ( ! setter )
  {
    throw NullPointerException ( JEM_FUNC,
                                 "NULL Field set function" );
  }
}


template <class G, class S>

  SettableFieldTemplate<G,S>::~SettableFieldTemplate ()

{}


//-----------------------------------------------------------------------
//   isStatic
//-----------------------------------------------------------------------


template <class G, class S>

  bool SettableFieldTemplate<G,S>::isStatic () const

{
  return FieldGetterTraits<G>::IS_STATIC;
}


//-----------------------------------------------------------------------
//   isSettable
//-----------------------------------------------------------------------


template <class G, class S>

  bool SettableFieldTemplate<G,S>::isSettable () const

{
  return true;
}


//-----------------------------------------------------------------------
//   getFieldType
//-----------------------------------------------------------------------


template <class G, class S>

  Class* SettableFieldTemplate<G,S>::getFieldType () const

{
  return FieldGetterTraits<G>::getFieldType ();
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


template <class G, class S>

  Ref<Object> SettableFieldTemplate<G,S>::get

  ( const Ref<Object>&  obj ) const

{
  return FieldGetterTraits<G>::get ( obj, getter_ );
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


template <class G, class S>

  void SettableFieldTemplate<G,S>::set

  ( const Ref<Object>&  obj,
    const Ref<Object>&  val ) const

{
  FieldSetterTraits<S>::set ( obj, setter_, val );
}



//=======================================================================
//   makeField
//=======================================================================


template <class Getter>

  inline Ref<Field>     makeField

  ( const String&         name,
    Getter                getter )

{
  return newInstance< FieldTemplate<Getter> > ( name, getter );
}


template <class Getter, class Setter>

  inline Ref<Field>     makeField

  ( const String&         name,
    Getter                getter,
    Setter                setter )

{
  return newInstance< SettableFieldTemplate<Getter,Setter> >

    ( name, getter, setter );
}


JEM_END_PACKAGE_BASE

#endif
