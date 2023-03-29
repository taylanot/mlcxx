
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

#ifndef JEM_IO_ARRAYINPUTSTREAM_H
#define JEM_IO_ARRAYINPUTSTREAM_H

#include <jem/base/array/Array.h>
#include <jem/io/InputStream.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class ArrayInputStream
//-----------------------------------------------------------------------


class ArrayInputStream : public InputStream
{
 public:

  typedef ArrayInputStream  Self;
  typedef InputStream       Super;


  explicit                  ArrayInputStream

    ( const Array<byte>&      buf );

  explicit                  ArrayInputStream

    ( Array<byte>&&           buf );

                            ArrayInputStream

    ( const Self&             rhs );

  virtual
    Ref<InputStream>        dup               () override;

  virtual idx_t             poll

    ( const Time&             timeout )          override;

  virtual idx_t             read

    ( void*                   buf,
      idx_t                   n )                override;

  virtual void              readAll

    ( void*                   buf,
      idx_t                   n )                override;

  virtual idx_t             skip

    ( idx_t                   n )                override;

  void                      reset             ();

  void                      reset

    ( const Array<byte>&      buf );

  void                      reset

    ( Array<byte>&&           buf );


 protected:

  virtual                  ~ArrayInputStream  ();


 private:

  Array<byte>               buf_;
  idx_t                     pos_;

};


JEM_END_PACKAGE( io )

#endif
