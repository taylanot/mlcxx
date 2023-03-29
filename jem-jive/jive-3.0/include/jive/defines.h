
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

#ifndef JIVE_DEFINES_H
#define JIVE_DEFINES_H

#include <jem/defines.h>


//-----------------------------------------------------------------------
//   namespace macros
//-----------------------------------------------------------------------


#define JIVE_BEGIN_PACKAGE( name )  namespace jive { namespace name {
#define JIVE_END_PACKAGE( name )    } }


//-----------------------------------------------------------------------
//   import some fundamental Jem classes
//-----------------------------------------------------------------------


namespace jem
{
  class Nil;
  class Object;
  class String;
  class NamedObject;

  template <class T> class Ref;
  template <>        class Ref<Object>;

  namespace mp
  {
    class Context;
  }

  namespace util
  {
    class Properties;
  }
}


namespace jive
{
  using jem::lint;
  using jem::ulint;
  using jem::idx_t;
  using jem::uidx_t;
  using jem::Nil;
  using jem::Ref;
  using jem::Object;
  using jem::String;
  using jem::NamedObject;
  using jem::util::Properties;

  typedef unsigned char     byte;
  typedef jem::mp::Context  MPContext;

  using namespace jem::literals;
}


#endif
