
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


#include <cstdlib>
#include <jem/base/limits.h>
#include <jem/base/Once.h>
#include <jem/base/Time.h>
#include <jem/base/array/utilities.h>
#include <jem/crypto/Random.h>
#include <jem/crypto/CryptoException.h>

extern "C"
{
  #include <openssl/rand.h>
}


#undef JEM_RDTSC
#undef JEM_RDTSC_T

#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))

extern "C"
{
  #include <x86intrin.h>
}

#  define JEM_RDTSC    __rdtsc
#  define JEM_RDTSC_T  unsigned long long

#elif defined (_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))

extern "C"
{
  #include <intrin.h>
}

#  define JEM_RDTSC    __rdtsc
#  define JEM_RDTSC_T  unsigned __int64

#endif


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class Random
//=======================================================================

//-----------------------------------------------------------------------
//   addBytes
//-----------------------------------------------------------------------


void Random::addBytes

  ( const ByteVector&  buf,
    double             entropy )

{
  addBytes ( makeContiguous( buf ).addr(), buf.size(), entropy );
}


void Random::addBytes

  ( const byte*  buf,
    idx_t        len,
    double       entropy )

{
  JEM_PRECHECK ( len     >= 0 );
  JEM_PRECHECK ( entropy >= 0.0 );

  const int  count = (int)

    jem::min ( (idx_t) maxOf<int>(), len );

  RAND_add ( buf, count, entropy );
}


//-----------------------------------------------------------------------
//   getBytes
//-----------------------------------------------------------------------


void Random::getBytes ( const ByteVector& buf )
{
  if ( ! findBytes( buf ) )
  {
    throw CryptoException (
      JEM_FUNC,
      "not enough random bytes available"
    );
  }
}


void Random::getBytes

  ( byte*  buf,
    idx_t  len )

{
  if ( ! findBytes( buf, len ) )
  {
    throw CryptoException (
      JEM_FUNC,
      "not enough random bytes available"
    );
  }
}


//-----------------------------------------------------------------------
//   findBytes
//-----------------------------------------------------------------------


bool Random::findBytes ( const ByteVector& buf )
{
  ByteVector  out = makeContiguous ( buf );
  bool        rv  = findBytes      ( out.addr(), out.size() );

  if ( rv && (out.addr() != buf.addr()) )
  {
    buf = out;
  }

  return rv;
}


bool Random::findBytes

  ( byte*  buf,
    idx_t  len )

{
  JEM_PRECHECK ( len >= 0 );

  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce_ ( once, init_ );

  if ( len > (idx_t) maxOf<int>() )
  {
    return false;
  }
  else
  {
    int  ierr = RAND_bytes ( buf, (int) len );

    return (ierr == 1 ? true : false);
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void Random::init_ ()
{
#ifdef JEM_RDTSC

  union
  {
    JEM_RDTSC_T  ticks;
    byte         bytes[sizeof(JEM_RDTSC_T)];
  }
  timeval;

  timeval.ticks = JEM_RDTSC ();

#else

  Time           now = Time::now ();

  union
  {
    lint         ticks;
    byte         bytes[sizeof(lint)];
  }
  timeval;

  timeval.ticks = (now.sec() ^ now.nsec());

#endif

  union
  {
    void*        addr;
    byte         bytes[sizeof(void*)];
  }
  addrval;

  int     n = (int) jem::min ( sizeof(addrval.bytes),
                               sizeof(timeval.bytes) );
  double  e = 0.0;

  addrval.addr = std::malloc ( 193 );

  for ( int i = 0; i < n; i++ )
  {
    timeval.bytes[i] ^= addrval.bytes[i];
  }

  n = (int) sizeof(timeval.bytes);

  if ( RAND_status() )
  {
    e = 0.5 * n;
  }
  else
  {
    e = (double) n;
  }

  RAND_add  ( timeval.bytes, n, e );
  std::free ( addrval.addr );
}


JEM_END_PACKAGE( crypto )

