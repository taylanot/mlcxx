
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

#ifndef JEM_IO_FILENAME_H
#define JEM_IO_FILENAME_H

#include <jem/base/String.h>
#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class FileName
//-----------------------------------------------------------------------


class FileName
{
 public:

  static char             fileSeparator   ();
  static char             pathSeparator   ();
  static char             suffixSeparator ();

  static bool             isAbsolute

    ( const String&         name  );

  static bool             isRoot

    ( const String&         name  );

  static String           getFileName

    ( const String&         name  );

  static String           getDirName

    ( const String&         name  );

  static String           getSuffix

    ( const String&         name  );

  static String           getBaseFileName

    ( const String&         name  );

  static String           getAbsoluteName

    ( const String&         name  );

  static String           getCanonicalName

    ( const String&         name  );

  static String           getParent

    ( const String&         name  );

  static String           translate

    ( const String&         name  );

  static String           simplify

    ( const String&         name  );

  static Array<String>    split

    ( const String&         name  );

  static String           join

    ( const String&         part1,
      const String&         part2 );

  static String           join

    ( const Array<String>&  parts );

  static idx_t            skipRoot

    ( const String&         name  );

  static idx_t            findSuffix

    ( const String&         name  );


 protected:

                          FileName ();
                         ~FileName ();


 private:

  static const char       PATH_SEP_;
  static const char       FILE_SEP_;
  static const char       SUFFIX_SEP_;

};


JEM_END_PACKAGE( io )

#endif
