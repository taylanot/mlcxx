
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


#include <jem/base/Once.h>
#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include <jem/base/Monitor.h>
#include <jem/base/LogBuffer.h>
#include <jem/util/List.h>
#include <jem/net/Socket.h>
#include <jem/net/Resolver.h>
#include <jem/net/TimeoutException.h>


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class ResolverQuery
//=======================================================================


class ResolverQuery : public Collectable
{
 public:

  typedef SocketAddress   Address;

  inline                  ResolverQuery

    ( Address::Kind         kind,
      Domain::Type          type,
      const String&         host,
      int                   port );


 public:

  const Address::Kind     kind;
  const Domain::Type      type;
  const String            host;
  const int               port;

  Monitor                 monitor;
  Ref<SocketAddress>      result;
  String                  error;
  bool                    failed;


 protected:

  virtual                ~ResolverQuery  ();

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline ResolverQuery::ResolverQuery

  ( Address::Kind  k,
    Domain::Type   t,
    const String&  n,
    int            p ) :

    kind ( k ),
    type ( t ),
    host ( n ),
    port ( p )

{
  failed = false;
}


ResolverQuery::~ResolverQuery ()
{}


//=======================================================================
//   class ResolverThread
//=======================================================================


class ResolverThread : public Thread
{
 public:

  typedef Thread          Super;
  typedef ResolverThread  Self;
  typedef ResolverQuery   Query;
  typedef SocketAddress   Address;


  virtual void            run             () override;

  Ref<Address>            lookup

    ( Address::Kind         kind,
      Domain::Type          type,
      const String&         host,
      int                   port,
      const Time&           timeout );

  static Self&            getInstance     ();


 protected:

  virtual                ~ResolverThread  ();


 private:

  void                    serveQuery_     ();

  static void             newInstance_    ();


 private:

  static Self*            instance_;

  Monitor                 monitor_;
  util::List<Ref<Query>>  queries_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


ResolverThread* ResolverThread::instance_ = nullptr;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


ResolverThread::~ResolverThread ()
{
  instance_ = nullptr;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void ResolverThread::run ()
{
  // This function should never return.

  allowCancel ( false );

  while ( true )
  {
    try
    {
      serveQuery_ ();
    }
    catch ( ... )
    {}
  }
}


//-----------------------------------------------------------------------
//   lookup
//-----------------------------------------------------------------------


Ref<SocketAddress> ResolverThread::lookup

  ( Address::Kind  kind,
    Domain::Type   type,
    const String&  host,
    int            port,
    const Time&    timeout )

{
  Ref<Query>    query  = newInstance<Query> ( kind, type,
                                              host, port );
  bool          failed = false;

  Ref<Address>  result;
  String        error;

  {
    Lock<Monitor>  lock ( monitor_ );

    queries_.pushFront ( query );
    monitor_.notify    ();
  }

  {
    Lock<Monitor>  lock ( query->monitor );

    if ( timeout < Time::zero() )
    {
      while ( ! (query->failed || query->result) )
      {
        query->monitor.wait ();
      }
    }
    else
    {
      Time  now = Time::now ();
      Time  end = now + timeout;

      while ( true )
      {
        if ( query->failed || query->result || (now > end) )
        {
          break;
        }

        query->monitor.wait ( end - now );

        now = Time::now ();
      }
    }

    result = query->result;
    error  = query->error;
    failed = query->failed;
  }

  if ( failed )
  {
    throw NetworkException (
      JEM_FUNC,
      error
    );
  }

  if ( ! result )
  {
    // Timeout occurred; remove the query from the list.

    Lock<Monitor>  lock ( monitor_ );

    for ( auto it = queries_.begin(); it != queries_.end(); ++it )
    {
      if ( *it == query )
      {
        queries_.erase ( it );
        break;
      }
    }

    throw TimeoutException (
      JEM_FUNC,
      "socket address lookup timed out"
    );
  }

  return result;
}


//-----------------------------------------------------------------------
//   getInstance
//-----------------------------------------------------------------------


ResolverThread& ResolverThread::getInstance ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, newInstance_ );

  return *instance_;
}


//-----------------------------------------------------------------------
//   serveQuery_
//-----------------------------------------------------------------------


void ResolverThread::serveQuery_ ()
{
  bool          failed = false;

  Ref<Address>  result;
  Ref<Query>    query;
  String        error;

  {
    Lock<Monitor>  lock ( monitor_ );

    while ( ! queries_.size() )
    {
      monitor_.wait ();
    }

    try
    {
      query = queries_.back ();

      queries_.popBack ();
    }
    catch ( ... )
    {
      query = nullptr;
    }
  }

  if ( ! query )
  {
    return;
  }

  try
  {
    LogBuffer::pushBack (
      JEM_FUNC,
      String::format (
        "resolving %s:%d",
        query->host,
        query->port
      )
    );

    result = SocketAddress::lookup (
      query->kind,
      query->type,
      query->host,
      query->port
    );
  }
  catch ( const Throwable& ex )
  {
    failed = true;
    error  = ex.what ();
  }
  catch ( ... )
  {
    failed = true;
  }

  {
    Lock<Monitor>  lock ( query->monitor );

    query->result = std::move ( result );
    query->error  = std::move ( error  );
    query->failed = failed;

    query->monitor.notify ();
  }
}


//-----------------------------------------------------------------------
//   newInstance_
//-----------------------------------------------------------------------


void ResolverThread::newInstance_ ()
{
  Ref<Self>  thread = newInstance<Self> ();

  thread->start ();

  Collectable::addRef ( *thread );

  instance_ = thread.get ();
}


//=======================================================================
//   class Resolver
//=======================================================================

//-----------------------------------------------------------------------
//   lookup
//-----------------------------------------------------------------------


Ref<SocketAddress> Resolver::lookup

  ( Address::Kind  kind,
    Domain::Type   type,
    const String&  host,
    int            port )

{
  return Address::lookup ( kind, type, host, port );
}


Ref<SocketAddress> Resolver::lookup

  ( Address::Kind  kind,
    Domain::Type   type,
    const String&  host,
    int            port,
    const Time&    timeout )

{
  ResolverThread&  thread = ResolverThread::getInstance ();

  return thread.lookup ( kind, type, host, port, timeout );
}


JEM_END_PACKAGE( net )

