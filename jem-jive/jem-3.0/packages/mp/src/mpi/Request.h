
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

#ifndef JEM_MP_MPI_REQUEST_H
#define JEM_MP_MPI_REQUEST_H

#include <jem/mp/mpi.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Request.h>


JEM_BEGIN_PACKAGE( mp )


class MPIContext;


JEM_BEGIN_NAMESPACE( mpi )


//-----------------------------------------------------------------------
//   class Request
//-----------------------------------------------------------------------


class Request : public mp::Request
{
 public:

  typedef Request           Self;
  typedef mp::Request       Super;


                            Request

    ( const Ref<MPIContext>&  ctx,
      const RecvBuffer&       buf,
      int                     src,
      int                     tag );

                            Request

    ( const Ref<MPIContext>&  ctx,
      const SendBuffer&       buf,
      int                     dest,
      int                     tag );

  virtual void              start   () override;

  virtual bool              test

    ( Status*                 stat )   override;

  virtual void              wait

    ( Status*                 stat )   override;

  virtual void              cancel  () override;


 protected:

  virtual                  ~Request ();


 private:

  Ref<MPIContext>           context_;
  MPI_Datatype              type_;
  MPI_Request               request_;
  bool                      active_;

};


JEM_END_NAMESPACE ( mpi )
JEM_END_PACKAGE   ( mp )

#endif
