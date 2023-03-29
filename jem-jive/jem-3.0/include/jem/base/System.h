
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

#ifndef JEM_BASE_SYSTEM_H
#define JEM_BASE_SYSTEM_H

#include <jem/io/Reader.h>
#include <jem/io/Writer.h>


namespace jem
{
  namespace io
  {
    class Logger;
  }
}


JEM_BEGIN_PACKAGE_BASE


class Throwable;


//-----------------------------------------------------------------------
//   class System
//-----------------------------------------------------------------------


class System
{
 public:

  typedef io::Writer      Writer;
  typedef io::Reader      Reader;
  typedef io::Logger      Logger;

  typedef void          (*SigHandler) ( int );


  enum                    OutputID
  {
                            ERR_STREAM,
                            WARN_STREAM,
                            OUT_STREAM
  };

  enum                  { LAST_OUTPUT = OUT_STREAM };


  static Reader&          in              ();
  static Writer&          err             ();
  static Writer&          warn            ();
  static Writer&          out             ();

  static Writer&          log             ();
  static Writer&          log

    ( const String&         tag );

  static Writer&          info            ();
  static Writer&          info

    ( const String&         tag );

  static Writer&          debug           ();
  static Writer&          debug

    ( const String&         tag );

  static void             flush           ();

  static Writer&          getOutput

    ( OutputID              oid );

  static void             setOutput

    ( OutputID              oid,
      const Ref<Writer>&    out );

  static void             setInStream

    ( const Ref<Reader>&    r );

  static void             setErrStream

    ( const Ref<Writer>&    w );

  static void             setWarnStream

    ( const Ref<Writer>&    w );

  static void             setOutStream

    ( const Ref<Writer>&    w );

  static Ref<Logger>      getLogger       ();

  static Ref<Logger>      setLogger

    ( const Ref<Logger>&    log );

  static String           getProperty

    ( const String&         key,
      const String&         def = "" );

  static String           getenv

    ( const String&         key,
      const String&         def = "" );

  static int              exec

    ( int                 (*func)() );

  static int              exec

    ( int                 (*func)( int, char** ),
      int                   argc,
      char**                argv );

  static void             werror

    ( const char*           str );

  static void             werror

    ( const char*           str,
      idx_t                 len );

  static SigHandler       setSigHandler

    ( int                   signum,
      SigHandler            handler );

  static void             abortOnSignal

    ( int                   signum );

  static String           getStackTrace

    ( int                   offset = 0 );

  static void             dumpStackTrace

    ( int                   offset = 0 )     noexcept;

  static void             printException

    ( Writer&               out,
      const Throwable&      ex );

  static void             printException

    ( Writer&               out,
      const String&         name  = "",
      const String&         where = "",
      const String&         what  = "",
      const String&         trace = "" );

  static void             printLogTrace

    ( Writer&               out );


 private:

  class                   Properties_;
  class                   IOStreams_;
  class                   Flusher_;

  friend class            IOStreams_;
  friend class            Flusher_;

};


JEM_END_PACKAGE_BASE

#endif
