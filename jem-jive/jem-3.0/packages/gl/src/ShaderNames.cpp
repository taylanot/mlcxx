
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


#include "ShaderNames.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//  class ShaderNames
//=======================================================================


const char*  ShaderNames::FLAT_CONST_COLOR  = "jemFlatConstColor";
const char*  ShaderNames::FLAT_MULTI_COLOR  = "jemFlatMultiColor";
const char*  ShaderNames::FLAT_TEX1D        = "jemFlatTex1D";
const char*  ShaderNames::FLAT_TEX1D_ALPHA  = "jemFlatTex1DAlpha";
const char*  ShaderNames::FLAT_TEX2D        = "jemFlatTex2D";
const char*  ShaderNames::FLAT_TEX2D_ALPHA  = "jemFlatTex2DAlpha";

const char*  ShaderNames::SHADE_CONST_COLOR = "jemShadeConstColor";
const char*  ShaderNames::SHADE_MULTI_COLOR = "jemShadeMultiColor";
const char*  ShaderNames::SHADE_TEX1D       = "jemShadeTex1D";
const char*  ShaderNames::SHADE_TEX1D_ALPHA = "jemShadeTex1DAlpha";
const char*  ShaderNames::SHADE_TEX2D       = "jemShadeTex2D";
const char*  ShaderNames::SHADE_TEX2D_ALPHA = "jemShadeTex2DAlpha";

const char*  ShaderNames::ACCUMULATION      = "jemAccumulation";
const char*  ShaderNames::ORBS_CHECKER      = "jemOrbsChecker";
const char*  ShaderNames::ORBS_UNIFORM      = "jemOrbsUniform";
const char*  ShaderNames::ORBS_VARYING      = "jemOrbsVarying";

const char*  ShaderNames::LINES_CONST_COLOR = "jemLinesConstColor";
const char*  ShaderNames::LINES_MULTI_COLOR = "jemLinesMultiColor";
const char*  ShaderNames::LINES_TEX1D       = "jemLinesTex1d";
const char*  ShaderNames::LINES_STIPPLE     = "jemLinesStipple";


//=======================================================================
//  class ShaderUnis
//=======================================================================

// Note that the strings in the class ShaderUnis are the exact names
// used as uniform variables in Jem shaders. See also the class
// ShaderSource.

const char*  ShaderUnis::ACC_FACTOR         = "uniAccFactor";
const char*  ShaderUnis::ALPHA_MIN          = "uniAlphaThreshold";
const char*  ShaderUnis::AMBIENT            = "uniAmbient";
const char*  ShaderUnis::COLOR              = "uniColor";
const char*  ShaderUnis::COLOR2             = "uniColor2";
const char*  ShaderUnis::FACTOR             = "uniFactor";
const char*  ShaderUnis::FRAME_SIZE         = "uniFrameSize";
const char*  ShaderUnis::LIGHT_COLOR0       = "uniLights[0].color";
const char*  ShaderUnis::LIGHT_COLOR1       = "uniLights[1].color";
const char*  ShaderUnis::LIGHT_COLOR2       = "uniLights[2].color";
const char*  ShaderUnis::LIGHT_COUNT        = "uniLightCount";
const char*  ShaderUnis::LIGHT_DIR0         = "uniLights[0].dir";
const char*  ShaderUnis::LIGHT_DIR1         = "uniLights[1].dir";
const char*  ShaderUnis::LIGHT_DIR2         = "uniLights[2].dir";
const char*  ShaderUnis::MVP                = "uniMvp";
const char*  ShaderUnis::MV                 = "uniMv";
const char*  ShaderUnis::NORM_MAT           = "uniNormMat";
const char*  ShaderUnis::PATTERN            = "uniPattern";
const char*  ShaderUnis::PROJ_MAT           = "uniProjMat";
const char*  ShaderUnis::RADIUS             = "uniRadius";
const char*  ShaderUnis::SHININESS          = "uniShininess";
const char*  ShaderUnis::SPECULAR           = "uniSpecular";
const char*  ShaderUnis::TEXTURE_1D         = "uniTex1D";
const char*  ShaderUnis::TEXTURE_2D         = "uniTex2D";
const char*  ShaderUnis::WIDTH              = "uniWidth";

const char*  ShaderUnis::LIGHT_COLOR[]      = { LIGHT_COLOR0,
                                                LIGHT_COLOR1,
                                                LIGHT_COLOR2 };

const char*  ShaderUnis::LIGHT_DIR[]        = { LIGHT_DIR0,
                                                LIGHT_DIR1,
                                                LIGHT_DIR2 };


//=======================================================================
//  class ShaderBuffers
//=======================================================================

// Note that the contents of the byte arrays are irrelevant;
// what matters is that each array has a unique address.

const byte   ShaderBuffers::COLORS     [1]  = { 42 };
const byte   ShaderBuffers::NORMALS    [1]  = { 42 };
const byte   ShaderBuffers::POSITIONS  [1]  = { 42 };
const byte   ShaderBuffers::RADII      [1]  = { 42 };
const byte   ShaderBuffers::TEX1_COORDS[1]  = { 42 };
const byte   ShaderBuffers::TEX2_COORDS[1]  = { 42 };


JEM_END_PACKAGE( gl )
