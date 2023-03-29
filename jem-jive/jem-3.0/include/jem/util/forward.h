
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

#ifndef JEM_UTIL_FORWARD_H
#define JEM_UTIL_FORWARD_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( util )


class                     BitSet;
class                     Calendar;
class                     CppScanner;
class                     CPUTimer;
class                     DictionaryEnumerator;
class                     Dictionary;
class                     Enumerable;
class                     Enumerator;
class                     FastAllocator;
class                     HashDictionary;
class                     MallocAllocator;
class                     MapEnumerator;
class                     None;
class                     ObjectCollection;
class                     ObjectConverter;
class                     ObjectHashMap;
class                     ObjectList;
class                     ObjectMap;
class                     ObjectPair;
class                     ObjectSequence;
class                     ObjFlex;
class                     Pattern;
class                     Properties;
class                     Random;
class                     SequenceEnumerator;
class                     StdAllocator;
class                     StringUtils;
class                     Timer;
class                     Tokenizer;
class                     WeakHashMap;

template <class T> class  Flex;
template <class T> class  Range;

typedef                   DictionaryEnumerator  DictEnum;
typedef                   Dictionary            Dict;
typedef                   HashDictionary        HashDict;
typedef                   MapEnumerator         MapEnum;
typedef                   ObjectCollection      ObjCollection;
typedef                   ObjectConverter       ObjConverter;
typedef                   ObjectHashMap         ObjHashMap;
typedef                   ObjectList            ObjList;
typedef                   ObjectMap             ObjMap;
typedef                   ObjectPair            ObjPair;
typedef                   ObjectSequence        ObjSequence;
typedef                   SequenceEnumerator    SeqEnum;


JEM_END_PACKAGE( util )

#endif
