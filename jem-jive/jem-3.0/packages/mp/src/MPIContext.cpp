
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


#include <jem/mp/config.h>

#ifdef JEM_USE_MPI

#include <jem/base/limits.h>
#include <jem/base/SerialSection.h>
#include <jem/mp/MPIContext.h>
#include "mpi/error.h"
#include "mpi/utilities.h"
#include "mpi/Request.h"
#include "mpi/RequestList.h"


JEM_BEGIN_PACKAGE( mp )


using mpi::TYPE_TABLE;
using mpi::convertType;


//=======================================================================
//   class MPIContext
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


MPI_Op  MPIContext::OP_TABLE_[OPCODE_COUNT];

bool    MPIContext::finalized_     = false;
int     MPIContext::instanceCount_ = 0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MPIContext::MPIContext ( MPI_Comm comm ) : comm_ ( comm )
{
  int  err;

  err = MPI_Comm_rank ( comm_, & myRank_ );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }

  err = MPI_Comm_size ( comm_, & size_ );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }
}


MPIContext::~MPIContext ()
{
  SerialSection  section;

  instanceCount_--;

  if ( ! finalized_ )
  {
    if ( comm_ != MPI_COMM_WORLD )
    {
      MPI_Comm_free ( & comm_ );
    }

    if ( instanceCount_ == 0 )
    {
      MPI_Finalize ();

      finalized_ = true;
    }
  }

  comm_ = MPI_COMM_NULL;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Ref<MPIContext> MPIContext::init ( int* argc, char*** argv )
{
  SerialSection  section;
  MPI_Comm       world;


  if ( instanceCount_ == 0 )
  {
    int  err;

    err = MPI_Init ( argc, argv );

    if ( err )
    {
      mpi::raiseError ( JEM_FUNC, err );
    }

    err = MPI_Comm_set_errhandler ( MPI_COMM_WORLD,
                                    MPI_ERRORS_RETURN );

    if ( err )
    {
      mpi::raiseError ( JEM_FUNC, err );
    }

    TYPE_TABLE[NO_TYPE]    = MPI_DATATYPE_NULL;
    TYPE_TABLE[BYTE]       = MPI_BYTE;
    TYPE_TABLE[CHAR]       = MPI_CHAR;
    TYPE_TABLE[SHORT]      = MPI_SHORT;
    TYPE_TABLE[INT]        = MPI_INT;
    TYPE_TABLE[LONG]       = MPI_LONG;
    TYPE_TABLE[LINT]       = MPI_LONG_LONG;

#ifdef JEM_INDEX_IS_INT
    TYPE_TABLE[IDX]        = MPI_INT;
#else
    TYPE_TABLE[IDX]        = TYPE_TABLE[LINT];
#endif

    TYPE_TABLE[FLOAT]      = MPI_FLOAT;
    TYPE_TABLE[DOUBLE]     = MPI_DOUBLE;
    TYPE_TABLE[INT_PAIR]   = MPI_2INT;
    TYPE_TABLE[DOUBLE_INT] = MPI_DOUBLE_INT;

    OP_TABLE_[SUM]         = MPI_SUM;
    OP_TABLE_[PROD]        = MPI_PROD;
    OP_TABLE_[MIN]         = MPI_MIN;
    OP_TABLE_[MAX]         = MPI_MAX;
    OP_TABLE_[BOR]         = MPI_BOR;
    OP_TABLE_[BAND]        = MPI_BAND;
    OP_TABLE_[LOR]         = MPI_LOR;
    OP_TABLE_[LAND]        = MPI_LAND;
  }

  finalized_ = false;

  instanceCount_++;

  world = MPI_COMM_WORLD;

  return newInstance<Self> ( world );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


int MPIContext::size () const
{
  return size_;
}


//-----------------------------------------------------------------------
//   myRank
//-----------------------------------------------------------------------


int MPIContext::myRank () const
{
  return myRank_;
}


//-----------------------------------------------------------------------
//   abort
//-----------------------------------------------------------------------


void MPIContext::abort ( int err )
{
  MPI_Abort ( MPI_COMM_WORLD, err );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Context> MPIContext::clone ()
{
  SerialSection  section;
  MPI_Comm       comm;
  int            err;

  err = MPI_Comm_dup ( comm_, & comm );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }

  instanceCount_++;

  return newInstance<Self> ( comm );
}


//-----------------------------------------------------------------------
//   getErrorString
//-----------------------------------------------------------------------


String MPIContext::getErrorString ( int err ) const
{
  return mpi::makeErrorString ( err );
}


//-----------------------------------------------------------------------
//   newRequestList
//-----------------------------------------------------------------------


Ref<RequestList> MPIContext::newRequestList ()
{
  return newInstance<mpi::RequestList> ( this );
}


//-----------------------------------------------------------------------
//   send
//-----------------------------------------------------------------------


void MPIContext::send

  ( const SendBuffer&  buf,
    int                dest,
    int                tag )

{
  if ( buf.size() > maxOf<int>() )
  {
    mpi::sizeOverflowError ( JEM_FUNC, buf.size() );
  }

  int  err = MPI_Send ( buf.addr(), (int) buf.size(),
                        convertType( buf.type() ),
                        dest, tag, comm_ );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   recv
//-----------------------------------------------------------------------


void MPIContext::recv

  ( const RecvBuffer&  buf,
    int                src,
    int                tag,
    Status*            stat )

{
  MPI_Datatype  type = convertType ( buf.type() );

  MPI_Status    status;
  int           size;


  if ( src == ANY_SOURCE )
  {
    src = MPI_ANY_SOURCE;
  }

  if ( tag == ANY_TAG )
  {
    tag = MPI_ANY_TAG;
  }

  if ( buf.size() > maxOf<int>() )
  {
    size = maxOf<int> ();
  }
  else
  {
    size = (int) buf.size ();
  }

  int  err = MPI_Recv ( buf.addr(), size, type,
                        src, tag, comm_, &status );

  if ( err )
  {
    if ( stat )
    {
      stat->error = status.MPI_ERROR;
    }

    mpi::raiseError ( JEM_FUNC, err );
  }

  if ( stat )
  {
    mpi::setStatus ( *stat, status, type );
  }
}


//-----------------------------------------------------------------------
//   initSend
//-----------------------------------------------------------------------


Ref<Request> MPIContext::initSend

  ( const SendBuffer&  buf,
    int                dest,
    int                tag )

{
  return newInstance<mpi::Request> ( this, buf, dest, tag );
}


//-----------------------------------------------------------------------
//   initRecv
//-----------------------------------------------------------------------


Ref<Request> MPIContext::initRecv

  ( const RecvBuffer&  buf,
    int                src,
    int                tag )

{
  return newInstance<mpi::Request> ( this, buf, src, tag );
}


//-----------------------------------------------------------------------
//   barrier
//-----------------------------------------------------------------------


void MPIContext::barrier ()
{
  int err = MPI_Barrier ( comm_ );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   broadcast
//-----------------------------------------------------------------------


void MPIContext::broadcast ( const SendBuffer& buf )
{
  if ( buf.size() > maxOf<int>() )
  {
    mpi::sizeOverflowError ( JEM_FUNC, buf.size() );
  }

  int  err = MPI_Bcast ( buf.addr (),
                         (int) buf.size (),
                         convertType ( buf.type() ),
                         myRank_, comm_ );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }
}


void MPIContext::broadcast ( const RecvBuffer& buf, int root )
{
  if ( buf.size() > maxOf<int>() )
  {
    mpi::sizeOverflowError ( JEM_FUNC, buf.size() );
  }

  int  err = MPI_Bcast ( buf.addr (),
                         (int) buf.size (),
                         convertType ( buf.type() ),
                         root, comm_ );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   reduce
//-----------------------------------------------------------------------


void MPIContext::reduce

  ( const RecvBuffer&  rbuf,
    const SendBuffer&  sbuf,
    int                root,
    Opcode             opcode )

{
  if ( rbuf.type() != sbuf.type() )
  {
    mpi::bufferTypeError ( JEM_FUNC, rbuf.type(), sbuf.type() );
  }

  if ( rbuf.size() > maxOf<int>() )
  {
    mpi::sizeOverflowError ( JEM_FUNC, rbuf.size() );
  }

  if ( rbuf.size() != sbuf.size() )
  {
    mpi::bufferSizeError ( JEM_FUNC, (int) rbuf.size(),
                                     (int) sbuf.size() );
  }

  int  err = MPI_Reduce ( sbuf.addr (),
                          rbuf.addr (),
                          (int) rbuf.size (),
                          convertType ( rbuf.type() ),
                          OP_TABLE_[opcode],
                          root,
                          comm_ );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   allreduce
//-----------------------------------------------------------------------


void MPIContext::allreduce

  ( const RecvBuffer&  rbuf,
    const SendBuffer&  sbuf,
    Opcode             opcode )

{
  if ( rbuf.type() != sbuf.type() )
  {
    mpi::bufferTypeError ( JEM_FUNC, rbuf.type(), sbuf.type() );
  }

  if ( rbuf.size() > maxOf<int>() )
  {
    mpi::sizeOverflowError ( JEM_FUNC, rbuf.size() );
  }

  if ( rbuf.size() != sbuf.size() )
  {
    mpi::bufferSizeError ( JEM_FUNC, (int) rbuf.size(),
                                     (int) sbuf.size() );
  }

  int  err = MPI_Allreduce ( sbuf.addr (),
                             rbuf.addr (),
                             (int) sbuf.size (),
                             convertType ( sbuf.type() ),
                             OP_TABLE_[opcode],
                             comm_ );

  if ( err )
  {
    mpi::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   finalize
//-----------------------------------------------------------------------


void MPIContext::finalize ()
{
  SerialSection  section;

  if ( ! finalized_ )
  {
    int  err = MPI_Finalize ();

    if ( err )
    {
      mpi::raiseError ( JEM_FUNC, err );
    }

    finalized_ = true;
  }
}


JEM_END_PACKAGE( mp )

#endif
