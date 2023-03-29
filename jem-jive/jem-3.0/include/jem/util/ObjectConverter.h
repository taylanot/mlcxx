
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

#ifndef JEM_UTIL_OBJECTCONVERTER_H
#define JEM_UTIL_OBJECTCONVERTER_H

#include <jem/base/Object.h>
#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ObjectConverter
//-----------------------------------------------------------------------


class ObjectConverter : public Object
{
 public:

  JEM_DECLARE_CLASS     ( ObjectConverter, Object );


  virtual void            convert

    ( bool&                 out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( char&                 out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( byte&                 out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( short&                out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( int&                  out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( long&                 out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( lint&                 out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( idx_t&                out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( float&                out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( double&               out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( String&               out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( Array<bool>&          out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( Array<int>&           out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( Array<idx_t>&         out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( Array<double>&        out,
      const Ref<Object>&    obj )              const = 0;

  virtual void            convert

    ( Array<String>&        out,
      const Ref<Object>&    obj )              const = 0;


 protected:

  virtual                ~ObjectConverter   ();

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef ObjectConverter   ObjConverter;


JEM_END_PACKAGE( util )

#endif
