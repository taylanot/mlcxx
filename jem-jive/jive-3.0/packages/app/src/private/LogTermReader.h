
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_APP_PRIVATE_LOGTERMREADER_H
#define JIVE_APP_PRIVATE_LOGTERMREADER_H

#include <jem/base/Clonable.h>
#include <jem/io/TermReader.h>


JIVE_BEGIN_PACKAGE( app )


using jem::Time;
using jem::io::TermReader;
using jem::io::WordCompleter;

class LogState;
class LogFileWriter;


//-----------------------------------------------------------------------
//   class LogTermReader
//-----------------------------------------------------------------------


class LogTermReader : public TermReader,
                      public jem::Clonable
{
 public:

  JEM_DECLARE_CLASS     ( LogTermReader, TermReader );


                          LogTermReader

    ( Ref<LogState>        state,
      Ref<LogFileWriter>   logger );

  virtual idx_t           poll

    ( const Time&           timeout )              override;

  virtual String          readLine

    ( const String&         prompt,
      WordCompleter*        wcomp )                override;

  virtual void            clearHistory    ()       override;
  virtual Ref<Object>     clone           () const override;


 protected:

  virtual                ~LogTermReader   ();


 private:

  void                    initReader_     ();


 private:

  Ref<LogState>           state_;
  Ref<LogFileWriter>      logger_;
  Ref<TermReader>         reader_;

};


JIVE_END_PACKAGE( app )

#endif
