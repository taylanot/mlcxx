
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

#ifndef JIVE_UTIL_FORWARD
#define JIVE_UTIL_FORWARD

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( util )


class                     AllItemGroup;
class                     ArrayItemGroup;
class                     ColoredItemGroup;
class                     Constraints;
class                     ConstraintsParser;
class                     Database;
class                     DatabaseSkin;
class                     DataParser;
class                     DBaseParser;
class                     DBasesParser;
class                     DenseDofSpace;
class                     DenseTable;
class                     DofSpace;
class                     DummyItemSet;
class                     EmptyItemGroup;
class                     EventMask;
class                     EventSource;
class                     FlexDofSpace;
class                     FloatDBColumn;
class                     FuncParser;
class                     FuncsParser;
class                     GroupConverter;
class                     GroupSet;
class                     GroupSetParser;
class                     HashItemMap;
class                     HeapItemMap;
class                     IncludeParser;
class                     IntDBColumn;
class                     ItemGroup;
class                     ItemGroupParser;
class                     ItemGroupSkin;
class                     ItemGroupsParser;
class                     ItemIDException;
class                     ItemMap;
class                     ItemSet;
class                     ItemSetSkin;
class                     MemberSet;
class                     MemberSetParser;
class                     ObjectConverter;
class                     PartParser;
class                     PointSet;
class                     PointSetParser;
class                     Random;
class                     Reordering;
class                     Runtime;
class                     SimpleItemMap;
class                     SparseDofSpace;
class                     SparseTable;
class                     StdDatabase;
class                     StdGroupSet;
class                     StdMemberSet;
class                     StdPointSet;
class                     TableException;
class                     Table;
class                     TableParser;
class                     TablesParser;
class                     UniformDofSpace;
class                     VectorManager;
class                     XDofSpace;
class                     XGroupSet;
class                     XItemGroup;
class                     XItemMap;
class                     XMemberSet;
class                     XPointSet;
class                     XTable;

typedef                   ObjectConverter    ObjConverter;
typedef                   ConstraintsParser  ConParser;


JEM_END_PACKAGE( util )

#endif
