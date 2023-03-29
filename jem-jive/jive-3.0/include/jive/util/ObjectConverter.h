
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

#ifndef JIVE_UTIL_OBJECTCONVERTER_H
#define JIVE_UTIL_OBJECTCONVERTER_H

#include <jem/util/Properties.h>
#include <jem/util/ObjectConverter.h>
#include <jive/Array.h>
#include <jive/util/import.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ObjectConverter
//-----------------------------------------------------------------------


class ObjectConverter : public jem::util::ObjConverter
{
 public:

  JEM_DECLARE_CLASS     ( ObjectConverter,
                          jem::util::ObjConverter );


  explicit                ObjectConverter

    ( const Properties&     globdat );

  virtual void            convert

    ( bool&                 out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( char&                 out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( byte&                 out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( short&                out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( int&                  out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( long&                 out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( lint&                 out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( idx_t&                out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( float&                out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( double&               out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( String&               out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( BoolVector&           out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( IntVector&            out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( IdxVector&            out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( Vector&               out,
      const Ref<Object>&    obj )              const override;

  virtual void            convert

    ( StringVector&         out,
      const Ref<Object>&    obj )              const override;


 protected:

  virtual                ~ObjectConverter   ();


 private:

  bool                    toBool_

    ( const String&         str )              const;

  lint                    toInt_

    ( const String&         str,
      lint                  minval,
      lint                  maxval )           const;

  double                  toFloat_

    ( const String&         expr,
      double                maxval )           const;

  static void             typeError_

    ( const String&         expected,
      const Ref<Object>&    obj );


 private:

  Properties              globdat_;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef ObjectConverter   ObjConverter;


JIVE_END_PACKAGE( util )

#endif
