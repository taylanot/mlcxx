
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

#ifndef JEM_GL_GLFUNCTABLES_H
#define JEM_GL_GLFUNCTABLES_H

#include <jem/base/Object.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


class GLFuncTable1Core;
class GLFuncTable1Compat;
class GLFuncTable1_1Core;
class GLFuncTable1_2Core;
class GLFuncTable1_3Core;
class GLFuncTable1_4Core;
class GLFuncTable1_5Core;
class GLFuncTable2Core;
class GLFuncTable2Compat;
class GLFuncTable2_1Core;
class GLFuncTable3Core;
class GLFuncTable3Compat;
class GLFuncTable3_1Core;
class GLFuncTable3_2Core;
class GLFuncTable3_3Core;
class GLFuncTable4Core;
class GLFuncTable4Compat;
class GLFuncTable4_1Core;
class GLFuncTable4_2Core;
class GLFuncTable4_3Core;
class GLFuncTable4_4Core;
class GLFuncTable4_5Core;


//-----------------------------------------------------------------------
//   class GLFuncTables
//-----------------------------------------------------------------------


class GLFuncTables
{
 public:

  template <class T>
    inline Ref<T>             get

    ( Ref<T>&                   table );


 public:

  GLFuncLoader                loader;
  Ref<GLFuncTable1Core>       funcs1Core;
  Ref<GLFuncTable1Compat>     funcs1Compat;
  Ref<GLFuncTable1_1Core>     funcs1_1Core;
  Ref<GLFuncTable1_2Core>     funcs1_2Core;
  Ref<GLFuncTable1_3Core>     funcs1_3Core;
  Ref<GLFuncTable1_4Core>     funcs1_4Core;
  Ref<GLFuncTable1_5Core>     funcs1_5Core;
  Ref<GLFuncTable2Core>       funcs2Core;
  Ref<GLFuncTable2Compat>     funcs2Compat;
  Ref<GLFuncTable2_1Core>     funcs2_1Core;
  Ref<GLFuncTable3Core>       funcs3Core;
  Ref<GLFuncTable3Compat>     funcs3Compat;
  Ref<GLFuncTable3_1Core>     funcs3_1Core;
  Ref<GLFuncTable3_2Core>     funcs3_2Core;
  Ref<GLFuncTable3_3Core>     funcs3_3Core;
  Ref<GLFuncTable4Core>       funcs4Core;
  Ref<GLFuncTable4Compat>     funcs4Compat;
  Ref<GLFuncTable4_1Core>     funcs4_1Core;
  Ref<GLFuncTable4_2Core>     funcs4_2Core;
  Ref<GLFuncTable4_3Core>     funcs4_3Core;
  Ref<GLFuncTable4_4Core>     funcs4_4Core;
  Ref<GLFuncTable4_5Core>     funcs4_5Core;

};



//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


template <class T>

  inline Ref<T> GLFuncTables::get ( Ref<T>& table )

{
  if ( ! table )
  {
    table = newInstance<T> ( &loader );
  }

  return table;
}


JEM_END_PACKAGE( gl )

#endif
