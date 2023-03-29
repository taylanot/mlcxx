
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

#ifndef IMPORT_H
#define IMPORT_H

#include <jive/Array.h>


namespace jem
{
  namespace util
  {
    class Properties;
  }
}

namespace jive
{
  namespace util
  {
    class XTable;
    class XDofSpace;
  }

  namespace algebra
  {
    class   MatrixBuilder;

    typedef MatrixBuilder  MBuilder;
  }

  namespace model
  {
    class Model;
  }

  namespace fem
  {
    class NodeSet;
    class ElementSet;
    class BezierElement;
  }
}


using namespace jem;

using jem::util::Properties;
using jive::Vector;
using jive::IdxVector;
using jive::Matrix;
using jive::IdxMatrix;
using jive::util::XTable;
using jive::util::XDofSpace;
using jive::algebra::MBuilder;
using jive::model::Model;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::fem::BezierElement;

#endif
