
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

#ifndef JIVE_MODEL_IMPORT_H
#define JIVE_MODEL_IMPORT_H

#include <jem/io/forward.h>
#include <jem/numeric/forward.h>
#include <jive/util/forward.h>
#include <jive/mp/forward.h>
#include <jive/algebra/forward.h>


JIVE_BEGIN_PACKAGE( model )


using jem::io::ObjectInput;
using jem::io::ObjectOutput;
using jem::io::Serializable;
using jem::numeric::Function;
using jive::util::Table;
using jive::util::ItemSet;
using jive::util::DofSpace;
using jive::mp::TableExchanger;
using jive::mp::VectorExchanger;
using jive::algebra::VectorSpace;
using jive::algebra::AbstractMatrix;
using jive::algebra::MatrixBuilder;


JIVE_END_PACKAGE( model )

#endif
