
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
#include <jem/mp/mpi.h>
#include <jem/mp/MPIContext.h>
#include "error.h"
#include "utilities.h"
#include "Request.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mpi )


//=======================================================================
//   class Request
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Request::Request

  ( const Ref<MPIContext>&  ctx,
    const RecvBuffer&       buf,
    int                     src,
    int                     tag ) :

    context_ ( ctx ),
    type_    ( convertType( buf.type() ) ),
    active_  ( false )

{
  if ( buf.size() > maxOf<int>() )
  {
    sizeOverflowError ( JEM_FUNC, buf.size() );
  }

  if ( src == ANY_SOURCE )
  {
    src = MPI_ANY_SOURCE;
  }

  if ( tag == ANY_TAG )
  {
    tag = MPI_ANY_TAG;
  }

  int  err = MPI_Recv_init ( buf.addr (),
                             (int) buf.size (),
                             type_,
                             src,
                             tag,
                             ctx->getComm (),
                             &request_ );

  if ( err )
  {
    raiseError ( JEM_FUNC, err );
  }
}


Request::Request

  ( const Ref<MPIContext>&  ctx,
    const SendBuffer&       buf,
    int                     dest,
    int                     tag ) :

    context_ ( ctx ),
    type_    ( convertType( buf.type() ) ),
    active_  ( false )

{
  if ( buf.size() > maxOf<int>() )
  {
    sizeOverflowError ( JEM_FUNC, buf.size() );
  }

  int  err = MPI_Send_init ( buf.addr (),
                             (int) buf.size (),
                             type_,
                             dest,
                             tag,
                             ctx->getComm (),
                             &request_ );

  if ( err )
  {
    raiseError ( JEM_FUNC, err );
  }
}


Request::~Request ()
{
  if ( ! MPIContext::finalized() )
  {
    if ( active_ )
    {
      MPI_Cancel ( & request_ );
    }

    MPI_Request_free ( & request_ );
  }
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void Request::start ()
{
  if ( ! active_ )
  {
    int  err = MPI_Start ( & request_ );

    if ( err )
    {
      raiseError ( JEM_FUNC, err );
    }

    active_ = true;
  }
}


//-----------------------------------------------------------------------
//   test
//-----------------------------------------------------------------------


bool Request::test ( Status* stat )
{
  MPI_Status  status;
  int         flag;


  if ( active_ )
  {
    int  err = MPI_Test ( & request_, & flag, & status );

    if ( err )
    {
      active_ = false;

      if ( stat )
      {
        stat->error = status.MPI_ERROR;
      }

      raiseError ( JEM_FUNC, err );
    }

    if ( flag )
    {
      active_ = false;

      if ( stat )
      {
        setStatus ( *stat, status, type_ );
      }
    }
  }
  else if ( stat )
  {
    *stat = EMPTY_STATUS;
  }

  return ! active_;
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


void Request::wait ( Status* stat )
{
  MPI_Status  status;

  if ( active_ )
  {
    int  err = MPI_Wait ( & request_, & status );

    active_ = false;

    if ( err )
    {
      if ( stat )
      {
        stat->error = status.MPI_ERROR;
      }

      raiseError ( JEM_FUNC, err );
    }

    if ( stat )
    {
      setStatus ( *stat, status, type_ );
    }
  }
  else if ( stat )
  {
    *stat = EMPTY_STATUS;
  }
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void Request::cancel ()
{
  MPI_Status  status;

  if ( active_ )
  {
    int  err = MPI_Cancel ( & request_ );

    if ( err )
    {
      raiseError ( JEM_FUNC, err );
    }

    active_ = false;
    err     = MPI_Wait ( & request_, & status );

    if ( err )
    {
      raiseError ( JEM_FUNC, err );
    }
  }
}


JEM_END_NAMESPACE ( mpi )
JEM_END_PACKAGE   ( mp )

#endif
