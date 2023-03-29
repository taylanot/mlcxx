
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

#ifndef JEM_BASE_CLASS_H
#define JEM_BASE_CLASS_H

#include <jem/base/array/Array.h>
#include <jem/base/Object.h>


namespace jem
{
  namespace util
  {
    class Dictionary;
  }
}


JEM_BEGIN_PACKAGE_BASE


class Field;
class Method;
class ClassTableTraits;
class AttributeTable;
class Constructor;


//-----------------------------------------------------------------------
//   class Class
//-----------------------------------------------------------------------


class Class
{
 public:

  typedef Constructor       Ctor;

  typedef Array
    < Ref<Ctor> >           CtorArray;
  typedef Array
    < Ref<Method> >         MethodArray;
  typedef Array
    < Ref<Field> >          FieldArray;

  typedef util::Dictionary  Dictionary;

  friend class              ClassTableTraits;


  String                    getName             () const noexcept;
  String                    getWhatis           () const;

  void                      setWhatis

    ( const String&           whatis );

  bool                      isBaseOf

    ( Class*                  type )               const;

  inline Class*             getSuperClass       () const noexcept;

  virtual bool              equalType

    ( const Ref<Object>&      obj )                const = 0;

  bool                      isInstance

    ( const Ref<Object>&      obj )                const;

  Ref<Object>               newInstance         () const;

  void                      addConstructor

    ( const Ref<Ctor>&        ctor );

  CtorArray                 getConstructors     () const;

  Ref<Ctor>                 getConstructor

    ( Class**                 argTypes,
      idx_t                   argCount );

  void                      addAttribute

    ( const String&           name,
      const Ref<Object>&      attrib );

  Ref<Dictionary>           getAttributes       () const;
  Ref<Dictionary>           getDeclaredAttribs  () const;

  Ref<Object>               getAttribute

    ( const String&           name )               const;

  Ref<Object>               getDeclaredAttrib

    ( const String&           name )               const;

  void                      addField

    ( const Ref<Field>&       field );

  FieldArray                getFields           () const;
  FieldArray                getDeclaredFields   () const;

  Ref<Field>                getField

    ( const String&           name )               const;

  Ref<Field>                getDeclaredField

    ( const String&           name )               const;

  void                      addMethod

    ( const Ref<Method>&      method );

  MethodArray               getMethods          () const;
  MethodArray               getDeclaredMethods  () const;

  Ref<Method>               getMethod

    ( const String&           name )               const;

  Ref<Method>               getDeclaredMethod

    ( const String&           name )               const;

  static void               registerClass

    ( Class*                  type );

  static Class*             findClass

    ( const String&           name );


 protected:

  explicit                  Class

    ( const String&           name );

                            Class

    ( Class*                  super,
      const String&           name );

  virtual                  ~Class               ();


 private:

                            Class

    ( const Class&            rhs );

  Class&                    operator =

    ( const Class&            rhs );

  void                      getAttributes_

    ( AttributeTable&         attribs )            const;


 private:

  class                     Data_;
  class                     StaticData_;

  Class*                    super_;
  String                    name_;
  Data_*                    data_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getSuperClass
//-----------------------------------------------------------------------


inline Class* Class::getSuperClass () const noexcept
{
  return super_;
}


JEM_END_PACKAGE_BASE

#endif
