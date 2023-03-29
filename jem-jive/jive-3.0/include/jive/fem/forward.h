
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

#ifndef JIVE_FEM_FORWARD_H
#define JIVE_FEM_FORWARD_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( fem )


class                     AutoBndGroup;
class                     AutoShapeTable1D;
class                     AutoShapeTable2D;
class                     AutoShapeTable3D;
class                     AutoShapeTable;
class                     BezierElement;
class                     BndgenModule;
class                     BndGroupConverter;
class                     BoundaryElement;
class                     BoundaryGroup;
class                     Boundary;
class                     BoundaryIterator;
class                     BoundaryManager;
class                     BoundarySet;
class                     CustomShapeTable;
class                     DataParser;
class                     ElementGroup;
class                     Element;
class                     ElementIterator;
class                     ElementSet;
class                     ElemGroupConverter;
class                     FEIterator;
class                     FEMatrixBuilder;
class                     GroupconvModule;
class                     IBoundaryIterator;
class                     InitModule;
class                     InputModule;
class                     InternalBoundary;
class                     InternalElement;
class                     MeshpartModule;
class                     MPInputModule;
class                     NodeGroup;
class                     NodeSet;
class                     OrderedBndGroup;
class                     PartitionModule;
class                     ShapeModule;
class                     SimpleBoundary;
class                     SimpleElement;
class                     VirtualBoundary;
class                     VirtualElement;
class                     VirtualIBoundary;
class                     XBoundarySet;
class                     XElementSet;
class                     XNodeSet;

typedef                   BoundaryElement       BElement;
typedef                   BoundaryIterator      BndIter;
typedef                   ElementIterator       ElemIter;
typedef                   IBoundaryIterator     IBndIter;
typedef                   InternalBoundary      IBoundary;
typedef                   InternalElement       IElement;


JEM_END_PACKAGE( fem )

#endif
