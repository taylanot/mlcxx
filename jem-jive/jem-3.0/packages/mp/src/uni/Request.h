
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

#ifndef JEM_MP_UNI_REQUEST_H
#define JEM_MP_UNI_REQUEST_H

#include <jem/mp/Buffer.h>
#include <jem/mp/Request.h>
#include "Message.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( uni )


class MessagePool;


//-----------------------------------------------------------------------
//   class Request
//-----------------------------------------------------------------------


class Request : public mp::Request
{
 public:

  typedef Request         Self;
  typedef mp::Request     Super;
  typedef MessagePool     MPool;


  enum Mode {
                          SEND_MODE,
                          RECV_MODE,
  };


  inline                  Request

    ( const SendBuffer&     buf,
      int                   tag,
      const Ref<MPool>&     pool )  noexcept;

  inline                  Request

    ( const RecvBuffer&     buf,
      int                   tag,
      const Ref<MPool>&     pool )  noexcept;

  virtual void            start  () override;

  virtual bool            test

    ( Status*               stat )  override;

  virtual void            wait

    ( Status*               stat )  override;

  virtual void            cancel () override;


 public:

  const Mode              mode;
  Ref<MPool>              mpool;
  Message                 message;
  bool                    active;


 protected:

  virtual                ~Request ();

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Request::Request

  ( const SendBuffer&  buf,
    int                tag,
    const Ref<MPool>&  pool ) noexcept :

    mode     ( SEND_MODE ),
    mpool    (      pool ),
    message  (  buf, tag ),
    active   (     false )

{}


inline Request::Request

  ( const RecvBuffer&  buf,
    int                tag,
    const Ref<MPool>&  pool ) noexcept :

    mode     ( RECV_MODE ),
    mpool    (      pool ),
    message  (  buf, tag ),
    active   (     false )

{}


JEM_END_NAMESPACE ( uni )
JEM_END_PACKAGE   ( mp )

#endif
