
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

#ifndef JEM_IO_OBJECTINPUT_H
#define JEM_IO_OBJECTINPUT_H

#include <jem/base/Object.h>
#include <jem/io/DataInput.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class ObjectInput
//-----------------------------------------------------------------------


class ObjectInput : public DataInput
{
 public:

  virtual Ref<Object>     decodeObject  () = 0;


 protected:

  virtual                ~ObjectInput   ();

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input, class T>

  inline void             decode

  ( Input&                  in,
    Ref<T>&                 obj );



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input, class T>

  inline void       decode

  ( Input&            in,
    Ref<T>&           obj )

{
  obj = checkedCast<T> ( in.decodeObject() );
}


template <class Input>

  inline void       decode

  ( Input&            in,
    Ref<Object>&      obj )

{
  obj = in.decodeObject ();
}


JEM_END_PACKAGE( io )

#endif
