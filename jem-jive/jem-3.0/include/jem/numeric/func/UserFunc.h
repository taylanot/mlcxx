
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

#ifndef JEM_NUMERIC_FUNC_USERFUNC_H
#define JEM_NUMERIC_FUNC_USERFUNC_H

#include <jem/base/Flags.h>
#include <jem/io/Serializable.h>
#include <jem/numeric/import.h>
#include <jem/numeric/func/Function.h>


JEM_BEGIN_PACKAGE( numeric )


namespace ufunc
{
  class CodeBlock;
}


//-----------------------------------------------------------------------
//   class UserFunc
//-----------------------------------------------------------------------


class UserFunc : public Function,
                 public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( UserFunc, Function );

  class                     ArgList;
  class                     NameSpace;

  enum                      ResolveOption
  {
                              RESOLVE_STRICT = 1 << 0
  };

  typedef Flags
    <ResolveOption>         ResolveOptions;


                            UserFunc    ();

  explicit                  UserFunc

    ( ufunc::CodeBlock*       code );

                            UserFunc

    ( const String&           args,
      const String&           expr );

  virtual String            toString    () const override;

  virtual void              readFrom

    ( ObjectInput&            in )               override;

  virtual void              writeTo

    ( ObjectOutput&           out )        const override;

  virtual idx_t             argCount    () const override;

  virtual double            getValue

    ( const double*           args )       const override;

  virtual double            getDeriv

    ( idx_t                   iarg,
      const double*           args )       const override;

  virtual idx_t             markUsedArgs

    ( bool*                   mask )       const override;

  void                      resolve

    ( ResolveOptions          options = 0 );

  void                      resolve

    ( const Dictionary&       symbols,
      ResolveOptions          options = 0 );

  void                      resolve

    ( const NameSpace&        symbols,
      ResolveOptions          options = 0 );

  void                      unresolve   ();

  static Ref<Self>          parse

    ( ArgList&                args,
      const String&           expr );


 protected:

  virtual                  ~UserFunc    ();


 private:

  ufunc::CodeBlock*         code_;

};


JEM_DEFINE_FLAG_OPS( UserFunc::ResolveOptions )


//-----------------------------------------------------------------------
//   class UserFunc::ArgList
//-----------------------------------------------------------------------


class UserFunc::ArgList : public Interface
{
 public:

  virtual idx_t             argCount    () const = 0;

  virtual idx_t             getArgIndex

    ( const String&           arg )              = 0;


 protected:

  virtual                  ~ArgList     ();

};


//-----------------------------------------------------------------------
//   class UserFunc::NameSpace
//-----------------------------------------------------------------------


class UserFunc::NameSpace : public Interface
{
 public:

  virtual bool              getVariable

    ( double&                 value,
      const String&           name )       const = 0;

  virtual Ref<Function>     getFunction

    ( const String&           name )       const = 0;

  static double             toFloat

    ( const Ref<Object>&      obj,
      const String&           name );


 protected:

  virtual                  ~NameSpace   ();

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


void                        parseUserFunc

  ( String&                   name,
    String&                   args,
    String&                   expr,
    const String&             input );


JEM_END_PACKAGE( numeric )

#endif
