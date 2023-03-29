
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

Array<double,2>  a ( 10, 10 );   // A (10 x 10) matrix.
Array<double>    b ( 10 );       // A vector with length 10.
double           x;

a      = 0.0;
a(2,2) = 1.0;
a(3,3) = 2.0;

a(slice(4,7),slice(4,7)) = 3.0;  // Set a sub-matrix to 3.0.

b = a(ALL,4);                    // Assign b the fifth column of a.
b = abs( b ) + 2.0;              // b[i] = abs( b[i] ) + 2.0.
x = max( b * b );                // x    = max( b[i] * b[i] ).
