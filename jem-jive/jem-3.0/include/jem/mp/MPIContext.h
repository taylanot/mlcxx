
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

#ifndef JEM_MP_MPICONTEXT_H
#define JEM_MP_MPICONTEXT_H

#include <jem/mp/mpi.h>
#include <jem/mp/Context.h>


JEM_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   class MPIContext
//-----------------------------------------------------------------------


class MPIContext : public Context
{
 public:

  typedef MPIContext        Self;
  typedef Context           Super;


  explicit                  MPIContext

    ( MPI_Comm                comm );

  static Ref<Self>          init

    ( int*                    argc,
      char***                 argv );

  virtual int               size           () const override;
  virtual int               myRank         () const override;

  virtual void              abort

    ( int                     err )                 override;

  virtual Ref<Context>      clone          ()       override;

  virtual String            getErrorString

    ( int                     err )           const override;

  virtual Ref<RequestList>  newRequestList ()       override;

  virtual void              send

    ( const SendBuffer&       buf,
      int                     dest,
      int                     tag )                 override;

  virtual void              recv

    ( const RecvBuffer&       buf,
      int                     src,
      int                     tag,
      Status*                 stat )                override;

  virtual Ref<Request>      initSend

    ( const SendBuffer&       buf,
      int                     dest,
      int                     tag )                 override;

  virtual Ref<Request>      initRecv

    ( const RecvBuffer&       buf,
      int                     src,
      int                     tag )                 override;

  virtual void              barrier ()              override;

  virtual void              broadcast

    ( const SendBuffer&       buf )                 override;

  virtual void              broadcast

    ( const RecvBuffer&       buf,
      int                     root )                override;

  virtual void              reduce

    ( const RecvBuffer&       rbuf,
      const SendBuffer&       sbuf,
      int                     root,
      Opcode                  opcode )              override;

  virtual void              allreduce

    ( const RecvBuffer&       rbuf,
      const SendBuffer&       sbuf,
      Opcode                  opcode )              override;

  inline MPI_Comm           getComm        () const noexcept;
  static void               finalize       ();
  inline static bool        finalized      ()       noexcept;


 protected:

  virtual                  ~MPIContext     ();


 private:

  static MPI_Op             OP_TABLE_[OPCODE_COUNT];

  static bool               finalized_;
  static int                instanceCount_;

  MPI_Comm                  comm_;
  int                       size_;
  int                       myRank_;

};


//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getComm
//-----------------------------------------------------------------------


inline MPI_Comm MPIContext::getComm () const noexcept
{
  return comm_;
}


//-----------------------------------------------------------------------
//   finalized
//-----------------------------------------------------------------------


inline bool MPIContext::finalized () noexcept
{
  return finalized_;
}


JEM_END_PACKAGE( mp )

#endif
