
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


#include <jem/gl/GLContext.h>
#include <jem/gl/GLException.h>
#include <jem/gl/ShaderProg.h>
#include "ShaderNames.h"
#include "ShaderSource.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   Vertex Shaders
//=======================================================================

//-----------------------------------------------------------------------
//   VERT_FLAT_CONST_COLOR_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_FLAT_CONST_COLOR_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;

uniform mat4  uniMvp;

void main ()
{
  gl_Position = uniMvp * vec4 ( inPos, 1 );
}

)glsl";


//-----------------------------------------------------------------------
//   VERT_FLAT_MULTI_COLOR_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_FLAT_MULTI_COLOR_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;
layout(location = 1) in vec4  inColor;

uniform mat4  uniMvp;

out     vec4  fragColor;

void main ()
{
  gl_Position = uniMvp * vec4 ( inPos, 1 );

  fragColor   = inColor;
}

)glsl";


//-----------------------------------------------------------------------
//    VERT_FLAT_TEX1D_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_FLAT_TEX1D_ = R"glsl(

#version 330 core

layout(location = 0) in vec3   inPos;
layout(location = 1) in float  inU;

uniform mat4   uniMvp;

out     float  U;

void main ()
{
  gl_Position = uniMvp * vec4 ( inPos, 1 );
  U           = inU;
}

)glsl";


//-----------------------------------------------------------------------
//   VERT_FLAT_TEX2D_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_FLAT_TEX2D_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;
layout(location = 1) in vec2  inUV;

uniform mat4  uniMvp;

out     vec2  UV;

void main ()
{
  gl_Position = uniMvp * vec4 ( inPos, 1 );
  UV          = inUV;
}

)glsl";


//-----------------------------------------------------------------------
//   VERT_SHADE_CONST_COLOR_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_SHADE_CONST_COLOR_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;
layout(location = 1) in vec3  inNormal;

uniform mat4  uniMv;
uniform mat4  uniProjMat;
uniform mat3  uniNormMat;

out struct    Data
{
  vec3          eye;
  vec3          normal;
}             data;

void main ()
{
  vec4  pos = uniMv * vec4 ( inPos, 1.0F );

  data.normal = normalize ( uniNormMat * inNormal );
  gl_Position = uniProjMat * pos;

  if ( abs( uniProjMat[3].w ) > 1E-6 )
  {
    // Orthogonal projection.

    data.eye = vec3 ( 0.0F, 0.0F, 1.0F );
  }
  else
  {
    // Perspective projection.

    data.eye = -normalize ( pos.xyz );
  }
}

)glsl";


//-----------------------------------------------------------------------
//   VERT_SHADE_MULTI_COLOR_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_SHADE_MULTI_COLOR_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;
layout(location = 1) in vec3  inNormal;
layout(location = 2) in vec4  inColor;

uniform mat4  uniMv;
uniform mat4  uniProjMat;
uniform mat3  uniNormMat;

out struct    Data
{
  vec3          eye;
  vec3          normal;
}             data;

out     vec4  fragColor;

void main ()
{
  vec4  pos = uniMv * vec4 ( inPos, 1.0F );

  data.normal = normalize ( uniNormMat * inNormal );
  fragColor   = inColor;
  gl_Position = uniProjMat * pos;

  if ( abs( uniProjMat[3].w ) > 1E-6 )
  {
    // Orthogonal projection.

    data.eye = vec3 ( 0.0F, 0.0F, 1.0F );
  }
  else
  {
    // Perspective projection.

    data.eye = -normalize ( pos.xyz );
  }
}

)glsl";


//-----------------------------------------------------------------------
//   VERT_SHADE_TEX1D_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_SHADE_TEX1D_ = R"glsl(

#version 330 core

layout(location = 0) in vec3   inPos;
layout(location = 1) in vec3   inNormal;
layout(location = 2) in float  inU;

uniform mat4   uniMv;
uniform mat4   uniProjMat;
uniform mat3   uniNormMat;

out struct     Data
{
  vec3           eye;
  vec3           normal;
}              data;

out     float  U;

void main ()
{
  vec4  pos = uniMv * vec4 ( inPos, 1.0F );

  data.normal = normalize ( uniNormMat * inNormal );
  gl_Position = uniProjMat * pos;
  U           = inU;

  if ( abs( uniProjMat[3].w ) > 1E-6 )
  {
    // Orthogonal projection.

    data.eye = vec3 ( 0.0F, 0.0F, 1.0F );
  }
  else
  {
    // Perspective projection.

    data.eye = -normalize ( pos.xyz );
  }
}

)glsl";


//-----------------------------------------------------------------------
//   VERT_SHADE_TEX2D_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_SHADE_TEX2D_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;
layout(location = 1) in vec3  inNormal;
layout(location = 2) in vec2  inUV;

uniform mat4  uniMv;
uniform mat4  uniProjMat;
uniform mat3  uniNormMat;

out struct    Data
{
  vec3          eye;
  vec3          normal;
}             data;

out     vec2  UV;

void main ()
{
  vec4  pos = uniMv * vec4 ( inPos, 1.0F );

  data.normal = normalize ( uniNormMat * inNormal );
  gl_Position = uniProjMat * pos;
  UV          = inUV;

  if ( abs( uniProjMat[3].w ) > 1E-6 )
  {
    // Orthogonal projection.

    data.eye = vec3 ( 0.0F, 0.0F, 1.0F );
  }
  else
  {
    // Perspective projection.

    data.eye = -normalize ( pos.xyz );
  }
}

)glsl";


//=======================================================================
//   Fragment Shaders
//=======================================================================

//-----------------------------------------------------------------------
//   FRAG_FLAT_CONST_COLOR_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_FLAT_CONST_COLOR_ = R"glsl(

#version 330 core

uniform vec4   uniColor;
uniform float  uniAlphaThreshold;

out     vec4   color;

void main()
{
  // Alpha testing

  if ( uniColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = uniColor;
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_FLAT_MULTI_COLOR_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_FLAT_MULTI_COLOR_ = R"glsl(

#version 330 core

uniform float  uniAlphaThreshold;

in      vec4   fragColor;

out     vec4   color;

void main()
{
  // Alpha testing

  if ( fragColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = fragColor;
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_FLAT_TEX1D_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_FLAT_TEX1D_ = R"glsl(

#version 330 core

uniform sampler1D  uniTex1D;
uniform float      uniAlphaThreshold;

in      float      U;

out     vec4       color;

void main()
{
  color = texture( uniTex1D, U );

  // Alpha testing

  if ( color.a < uniAlphaThreshold )
  {
    discard;
  }
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_FLAT_TEX1D_ALPHA_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_FLAT_TEX1D_ALPHA_ = R"glsl(

#version 330 core

uniform vec4       uniColor;
uniform sampler1D  uniTex1D;
uniform float      uniAlphaThreshold;

in      float      U;

out     vec4       color;

void main()
{
  color.rgb = uniColor.rgb;
  color.a   = texture( uniTex1D, U ).a;

  // Alpha testing

  if ( color.a < uniAlphaThreshold )
  {
    discard;
  }
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_FLAT_TEX2D_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_FLAT_TEX2D_ = R"glsl(

#version 330 core

uniform sampler2D  uniTex2D;
uniform float      uniAlphaThreshold;

in      vec2       UV;

out     vec4       color;

void main()
{
  color = texture( uniTex2D, UV );

  // Alpha testing

  if ( color.a < uniAlphaThreshold )
  {
    discard;
  }
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_FLAT_TEX2D_ALPHA_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_FLAT_TEX2D_ALPHA_ = R"glsl(

#version 330 core

uniform vec4       uniColor;
uniform sampler2D  uniTex2D;
uniform float      uniAlphaThreshold;

in      vec2       UV;

out     vec4       color;

void main()
{
  color.rgb = uniColor.rgb;
  color.a   = texture( uniTex2D, UV ).a;

  // Alpha testing

  if ( color.a < uniAlphaThreshold )
  {
    discard;
  }
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_SHADE_CONST_COLOR_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_SHADE_CONST_COLOR_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform vec4       uniColor;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform float      uniAlphaThreshold;

in struct          Data
{
  vec3               eye;
  vec3               normal;
}                  data;

out     vec4       color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  vec3   n         = gl_FrontFacing ? normal : -normal;
  float  intensity = clamp ( -dot( n, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, n ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
//   main()
//-----------------------------------------------------------------------


void main()
{
  // Alpha testing

  if ( uniColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = vec4 ( 0.0, 0.0, 0.0, uniColor.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( uniColor, data.normal, data.eye,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * uniColor.xyz );
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_SHADE_MULTI_COLOR_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_SHADE_MULTI_COLOR_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform float      uniAlphaThreshold;

in struct          Data
{
  vec3               eye;
  vec3               normal;
}                  data;

in      vec4       fragColor;

out     vec4       color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  vec3   n         = gl_FrontFacing ? normal : -normal;
  float  intensity = clamp ( -dot( n, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, n ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
//   main()
//-----------------------------------------------------------------------


void main()
{
  // Alpha testing

  if ( fragColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = vec4 ( 0.0, 0.0, 0.0, fragColor.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( fragColor, data.normal, data.eye,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * fragColor.xyz );
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_SHADE_TEX1D_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_SHADE_TEX1D_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform sampler1D  uniTex1D;
uniform float      uniAlphaThreshold;

in struct          Data
{
  vec3               eye;
  vec3               normal;
}                  data;

in      float      U;

out     vec4       color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  vec3   n         = gl_FrontFacing ? normal : -normal;
  float  intensity = clamp ( -dot( n, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, n ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
//   main()
//-----------------------------------------------------------------------


void main()
{
  vec4  fragColor = texture ( uniTex1D, U );

  // Alpha testing

  if ( fragColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = vec4 ( 0.0, 0.0, 0.0, fragColor.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( fragColor, data.normal, data.eye,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * fragColor.xyz );
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_SHADE_TEX1D_ALPHA_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_SHADE_TEX1D_ALPHA_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform vec4       uniColor;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform sampler1D  uniTex1D;
uniform float      uniAlphaThreshold;

in struct          Data
{
  vec3               eye;
  vec3               normal;
}                  data;

in      float      U;

out     vec4       color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  vec3   n         = gl_FrontFacing ? normal : -normal;
  float  intensity = clamp ( -dot( n, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, n ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
//   main()
//-----------------------------------------------------------------------


void main()
{
  vec4   fragColor = vec4 ( uniColor.rgb, texture( uniTex1D, U ).a );

  // Alpha testing

  if ( fragColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = vec4 ( 0.0, 0.0, 0.0, fragColor.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( fragColor, data.normal, data.eye,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * fragColor.xyz );
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_SHADE_TEX2D_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_SHADE_TEX2D_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform sampler2D  uniTex2D;
uniform float      uniAlphaThreshold;

in struct          Data
{
  vec3               eye;
  vec3               normal;
}                  data;

in      vec2       UV;

out     vec4       color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  vec3   n         = gl_FrontFacing ? normal : -normal;
  float  intensity = clamp ( -dot( n, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, n ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
//   main()
//-----------------------------------------------------------------------


void main()
{
  vec4  fragColor = texture( uniTex2D, UV );

  // Alpha testing

  if ( fragColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = vec4 ( 0.0, 0.0, 0.0, fragColor.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( fragColor, data.normal, data.eye,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * fragColor.xyz );
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_SHADE_TEX2D_ALPHA_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_SHADE_TEX2D_ALPHA_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform vec4       uniColor;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform sampler2D  uniTex2D;
uniform float      uniAlphaThreshold;

in struct          Data
{
  vec3               eye;
  vec3               normal;
}                  data;

in      vec2       UV;

out     vec4       color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  vec3   n         = gl_FrontFacing ? normal : -normal;
  float  intensity = clamp ( -dot( n, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, n ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
//   main()
//-----------------------------------------------------------------------


void main()
{
  vec4  fragColor = vec4 ( uniColor.rgb, texture( uniTex2D, UV ).a );

  // Alpha testing

  if ( fragColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = vec4 ( 0.0, 0.0, 0.0, fragColor.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( fragColor, data.normal, data.eye,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * fragColor.xyz );
}

)glsl";


//=======================================================================
//   Accumulation Shaders
//=======================================================================

//-----------------------------------------------------------------------
//   VERT_CONST_ACCUMULATION_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_CONST_ACCUMULATION_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;

out                     vec2  UV;

void main ()
{
  gl_Position = vec4 ( inPos, 1.0F );
  UV          = 0.5F*(inPos.xy + vec2(1.0F,1.0F));
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_CONST_ACCUMULATION_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_CONST_ACCUMULATION_ = R"glsl(

#version 330 core

uniform sampler2D  uniTex2D;
uniform float      uniAccFactor;

in      vec2       UV;

out     vec4       color;

void main()
{
  color = vec4 ( texture( uniTex2D, UV ).rgb, uniAccFactor );
}

)glsl";


//=======================================================================
//    Orb Shader
//=======================================================================

//-----------------------------------------------------------------------
//   VERT_UNIFORM_ORB_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_UNIFORM_ORB_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;

uniform mat4  uniMv;

out struct    GeomData
{
  vec3          center;
}             geomData;

void main()
{
  geomData.center = (uniMv * vec4( inPos, 1.0F )).xyz;
  gl_Position     = vec4 ( inPos, 1.0F );
}

)glsl";


//-----------------------------------------------------------------------
//  GEOM_UNIFORM_ORB_
//-----------------------------------------------------------------------


const char* ShaderSource::GEOM_UNIFORM_ORB_ = R"glsl(

#version 330 core

layout(points)                         in;
layout(triangle_strip, max_vertices=4) out;

uniform  mat4   uniProjMat;
uniform  mat3   uniNormMat;
uniform  float  uniRadius;

in struct       GeomData
{
  vec3            center;
}               geomData[];

flat out struct FragData
{
  vec3            center;
}               fragData;

out      vec3   surfPoint;


//-----------------------------------------------------------------------
// main()
//-----------------------------------------------------------------------


void main()
{
  vec3   center =  geomData[0].center;

  vec4   w0, w1, w2, w3;
  vec3   up, right;

  up    = vec3 ( 0.0F, uniRadius, 0.0F );
  right = vec3 ( uniRadius, 0.0F, 0.0F );
  w0    = vec4 ( center - right - up, 1.0F );
  w1    = vec4 ( center + right - up, 1.0F );
  w2    = vec4 ( center - right + up, 1.0F );
  w3    = vec4 ( center + right + up, 1.0F );

  fragData.center = geomData[0].center;

  // Vertex 1
  surfPoint   = w0.xyz;
  gl_Position = uniProjMat * w0;
  EmitVertex();

  // Vertex 2
  surfPoint   = w1.xyz;
  gl_Position = uniProjMat * w1;
  EmitVertex();

  // Vertex 3
  surfPoint   = w2.xyz;
  gl_Position = uniProjMat * w2;
  EmitVertex();

  // Vertex 4
  surfPoint   = w3.xyz;
  gl_Position = uniProjMat * w3;
  EmitVertex();

  EndPrimitive();
}

)glsl";


//-----------------------------------------------------------------------
//  FRAG_UNIFORM_ORB_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_UNIFORM_ORB_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform mat4       uniProjMat;
uniform vec4       uniColor;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform float      uniAlphaThreshold;
uniform float      uniRadius;

flat in struct     FragData
{
  vec3               center;
}                  fragData;

in      vec3       surfPoint;

out     vec4       color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  float  intensity = clamp ( -dot( normal, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, normal ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
//  main()
//-----------------------------------------------------------------------


void main()
{
  vec3   rayOrig = surfPoint;
  vec3   center  = fragData.center;
  float  r2      = uniRadius*uniRadius;

  vec3   rayDir;
  vec3   pointHit;
  vec3   normal;
  vec3   diff;
  vec2   clip;
  float  d2;
  float  dist;


  // Alpha testing

  if ( uniColor.a < uniAlphaThreshold )
  {
    discard;
  }

  if ( abs( uniProjMat[3].w ) > 1E-6 )
  {
    // Orthogonal projection.

    rayDir = vec3 ( 0.0F, 0.0F, 1.0F );
  }
  else
  {
    // Perspective projection.

    rayDir = -normalize ( rayOrig );
  }

  diff = rayOrig - center;
  d2   = dot ( diff, diff );

  if ( d2 > r2 )
  {
    discard;
  }

  dist         = sqrt ( r2 - d2 );
  pointHit     = rayOrig + dist*rayDir;
  normal       = normalize ( pointHit - center );
  clip         = pointHit.z*uniProjMat[2].zw + uniProjMat[3].zw;
  gl_FragDepth = 0.5F + 0.5F*(clip.x / clip.y);
  color        = vec4 ( 0.0, 0.0, 0.0, uniColor.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( uniColor, normal, rayDir,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * uniColor.xyz );
}

)glsl";


//-----------------------------------------------------------------------
//   VERT_VARYING_ORB_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_VARYING_ORB_ = R"glsl(

#version 330 core

layout(location = 0) in vec3   inPos;
layout(location = 1) in vec4   inColor;
layout(location = 2) in float  inRadius;

uniform mat4  uniMv;

out struct    GeomData
{
  vec3          center;
  vec4          color;
  float         radius;
}             geomData;

void main()
{
  geomData.center = (uniMv * vec4( inPos, 1.0F )).xyz;
  geomData.color  = inColor;
  geomData.radius = inRadius;
  gl_Position     = vec4 ( inPos, 1.0F );
}

)glsl";


//-----------------------------------------------------------------------
//  GEOM_VARYING_ORB_
//-----------------------------------------------------------------------


const char* ShaderSource::GEOM_VARYING_ORB_ = R"glsl(

#version 330 core

layout(points)                         in;
layout(triangle_strip, max_vertices=4) out;

uniform  mat4  uniProjMat;

in struct      GeomData
{
  vec3           center;
  vec4           color;
  float          radius;
}              geomData[];

flat out struct FragData
{
  vec3            center;
  vec4            color;
  float           radius;
}              fragData;

out vec3       surfPoint;


//-----------------------------------------------------------------------
// main()
//-----------------------------------------------------------------------


void main()
{
  vec3   center =  geomData[0].center;
  float  r      =  geomData[0].radius;

  vec4   w0, w1, w2, w3;
  vec3   up, right;

  up    = vec3 ( 0.0F, r, 0.0F );
  right = vec3 ( r, 0.0F, 0.0F );
  w0    = vec4 ( center - right - up, 1.0F );
  w1    = vec4 ( center + right - up, 1.0F );
  w2    = vec4 ( center - right + up, 1.0F );
  w3    = vec4 ( center + right + up, 1.0F );

  fragData.center = geomData[0].center;
  fragData.color  = geomData[0].color;
  fragData.radius = geomData[0].radius;

  // Vertex 1
  surfPoint   = w0.xyz;
  gl_Position = uniProjMat * w0;
  EmitVertex();

  // Vertex 2
  surfPoint   = w1.xyz;
  gl_Position = uniProjMat * w1;
  EmitVertex();

  // Vertex 3
  surfPoint   = w2.xyz;
  gl_Position = uniProjMat * w2;
  EmitVertex();

  // Vertex 4
  surfPoint   = w3.xyz;
  gl_Position = uniProjMat * w3;
  EmitVertex();

  EndPrimitive();
}

)glsl";


//-----------------------------------------------------------------------
//  FRAG_VARYING_ORB_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_VARYING_ORB_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform mat4       uniProjMat;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform float      uniAlphaThreshold;

flat in struct     FragData
{
  vec3               center;
  vec4               color;
  float              radius;
}                  fragData;

in  vec3           surfPoint;

out vec4           color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  float  intensity = clamp ( -dot( normal, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, normal ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
// main()
//-----------------------------------------------------------------------


void main()
{
  vec3   rayOrig = surfPoint;
  vec3   center  = fragData.center;
  float  r2      = fragData.radius*fragData.radius;

  vec3   rayDir;
  vec3   pointHit;
  vec3   normal;
  vec3   diff;
  vec2   clip;
  float  d2;
  float  dist;


  // Alpha testing

  if ( fragData.color.a < uniAlphaThreshold )
  {
    discard;
  }

  if ( abs( uniProjMat[3].w ) > 1E-6 )
  {
    // Orthogonal projection.

    rayDir = vec3 ( 0.0F, 0.0F, 1.0F );
  }
  else
  {
    // Perspective projection.

    rayDir = -normalize ( rayOrig );
  }

  diff = rayOrig - center;
  d2   = dot ( diff, diff );

  if ( d2 > r2 )
  {
    discard;
  }

  dist         = sqrt ( r2 - d2 );
  pointHit     = rayOrig + dist*rayDir;
  normal       = normalize ( pointHit - center );
  clip         = pointHit.z*uniProjMat[2].zw + uniProjMat[3].zw;
  gl_FragDepth = 0.5F + 0.5F*(clip.x / clip.y);
  color        = vec4 ( 0.0, 0.0, 0.0, fragData.color.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( fragData.color, normal, rayDir,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * fragData.color.xyz );
}

)glsl";


//-----------------------------------------------------------------------
//  GEOM_CHECKER_ORB_
//-----------------------------------------------------------------------


const char* ShaderSource::GEOM_CHECKER_ORB_ = R"glsl(

#version 330 core

layout(points)                         in;
layout(triangle_strip, max_vertices=4) out;

uniform  mat4   uniProjMat;
uniform  mat3   uniNormMat;
uniform  float  uniRadius;

in struct       GeomData
{
  vec3            center;
}               geomData[];

flat out struct FragData
{
  vec3            center;
  vec3            xAxis;
  vec3            yAxis;
  vec3            zAxis;
}               fragData;

out vec3        surfPoint;


//-----------------------------------------------------------------------
// main()
//-----------------------------------------------------------------------


void main()
{
  vec3  center = geomData[0].center;

  vec4  w0, w1, w2, w3;
  vec3  up, right;

  up    = vec3 ( 0.0F, uniRadius, 0.0F );
  right = vec3 ( uniRadius, 0.0F, 0.0F );
  w0    = vec4 ( center - right - up, 1.0F );
  w1    = vec4 ( center + right - up, 1.0F );
  w2    = vec4 ( center - right + up, 1.0F );
  w3    = vec4 ( center + right + up, 1.0F );

  fragData.center = geomData[0].center;

  // Calculate orienation axis in ModelView. These are used to make the
  // checkerboard pattern on the sphere.

  fragData.xAxis  = uniNormMat * vec3( 1, 0, 0 );
  fragData.yAxis  = uniNormMat * vec3( 0, 1, 0 );
  fragData.zAxis  = uniNormMat * vec3( 0, 0, 1 );

  // Vertex 1
  surfPoint   = w0.xyz;
  gl_Position = uniProjMat * w0;
  EmitVertex();

  // Vertex 2
  surfPoint   = w1.xyz;
  gl_Position = uniProjMat * w1;
  EmitVertex();

  // Vertex 3
  surfPoint   = w2.xyz;
  gl_Position = uniProjMat * w2;
  EmitVertex();

  // Vertex 4
  surfPoint   = w3.xyz;
  gl_Position = uniProjMat * w3;
  EmitVertex();

  EndPrimitive();
}

)glsl";


//-----------------------------------------------------------------------
//  FRAG_CHECKER_ORB_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_CHECKER_ORB_ = R"glsl(

#version 330 core

#define ALMOST_ZERO  1.0E-8

struct             LightInfo
{
  vec3               dir;
  vec3               color;
};

uniform LightInfo  uniLights[3];
uniform int        uniLightCount;
uniform mat4       uniProjMat;
uniform vec4       uniColor;
uniform vec4       uniColor2;
uniform vec3       uniAmbient;
uniform vec3       uniSpecular;
uniform float      uniShininess;
uniform float      uniAlphaThreshold;
uniform float      uniRadius;

flat in struct     FragData
{
  vec3               center;
  vec3               xAxis;
  vec3               yAxis;
  vec3               zAxis;
}                  fragData;

in      vec3       surfPoint;

out     vec4       color;


//-----------------------------------------------------------------------
//  computeLight()
//-----------------------------------------------------------------------


vec3 computeLight

  ( vec4  objColor,
    vec3  normal,
    vec3  eye,
    vec3  lightDir,
    vec3  lightColor )

{
  vec3   spec      = vec3  ( 0.0 );
  float  intensity = clamp ( -dot( normal, lightDir ), 0.0F, 1.0F );

  if ( dot( uniSpecular, vec3( 1.0F ) ) > ALMOST_ZERO )
  {
    vec3   h       = normalize ( eye - lightDir );
    float  intSpec = max ( dot( h, normal ), 0.0F );

    spec = uniSpecular * pow ( intSpec, uniShininess );
  }

  return intensity * lightColor * objColor.xyz + spec;
}


//-----------------------------------------------------------------------
// main()
//-----------------------------------------------------------------------


void main()
{
  vec3   rayOrig = surfPoint;
  vec3   center  = fragData.center;
  float  r2      = uniRadius*uniRadius;

  vec3   rayDir;
  vec3   pointHit;
  vec3   normal;
  vec3   diff;
  vec4   partColor;
  vec2   clip;
  float  d2;
  float  dist;


  if ( abs( uniProjMat[3].w ) > 1E-6 )
  {
    // Orthogonal projection.

    rayDir = vec3 ( 0.0F, 0.0F, 1.0F );
  }
  else
  {
    // Perspective projection.

    rayDir = -normalize ( rayOrig );
  }

  diff = rayOrig - center;
  d2   = dot ( diff, diff );

  if ( d2 > r2 )
  {
    discard;
  }

  dist         = sqrt ( r2 - d2 );
  pointHit     = rayOrig + dist*rayDir;
  normal       = normalize ( pointHit - center );
  clip         = pointHit.z*uniProjMat[2].zw + uniProjMat[3].zw;
  gl_FragDepth = 0.5F + 0.5F*(clip.x / clip.y);
  partColor    =

    ( (dot( normal, fragData.xAxis ) *
       dot( normal, fragData.yAxis ) *
       dot( normal, fragData.zAxis )) < 0.0F ) ?
      uniColor :
      uniColor2;

  // Alpha testing

  if ( partColor.a < uniAlphaThreshold )
  {
    discard;
  }

  color = vec4 ( 0.0, 0.0, 0.0, partColor.a );

  for ( int i = 0; i < uniLightCount; i++ )
  {
    color.xyz = color.xyz +
                computeLight( partColor, normal, rayDir,
                              uniLights[i].dir, uniLights[i].color );
  }

  color.xyz = max( color.xyz, uniAmbient * partColor.xyz );
}

)glsl";


//=======================================================================
//    Line Shader
//=======================================================================

// The line geometry shaders create a triangle-strip from lines. The
// frame size is required to determine the size of a pixel in relative
// coordinates.

//-----------------------------------------------------------------------
//   VERT_LINES_MULTI_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_LINES_MULTI_ = R"glsl(

#version 330 core

layout(location = 0) in vec3  inPos;
layout(location = 1) in vec4  inColor;

uniform mat4  uniMvp;

out     vec4  geomColor;

void main ()
{
  gl_Position = uniMvp * vec4 ( inPos, 1 );

  geomColor   = inColor;
}

)glsl";


//-----------------------------------------------------------------------
//   VERT_LINES_TEX1D_
//-----------------------------------------------------------------------


const char* ShaderSource::VERT_LINES_TEX1D_ = R"glsl(

#version 330 core

layout(location = 0) in vec3   inPos;
layout(location = 1) in float  inVxColor;

uniform mat4   uniMvp;

out     float  geomVxColor;

void main ()
{
  gl_Position = uniMvp * vec4 ( inPos, 1 );
  geomVxColor = inVxColor;
}

)glsl";


//-----------------------------------------------------------------------
//   GEOM_LINES_CONST_
//-----------------------------------------------------------------------


const char* ShaderSource::GEOM_LINES_CONST_ = R"glsl(

#version 330 core

layout(lines)                           in;
layout(triangle_strip, max_vertices=4)  out;

uniform vec2   uniFrameSize;
uniform float  uniWidth;


void main()
{
  vec4  pos0 = gl_in[0].gl_Position;
  vec4  pos1 = gl_in[1].gl_Position;
  vec2  dir  = normalize ( (pos1.xy / pos1.w) - (pos0.xy / pos0.w) );
  vec2  up   = uniWidth * vec2(  dir.y / uniFrameSize.x,
                                -dir.x / uniFrameSize.y );

  vec2  offset;


  offset         = pos0.w  * up;

  gl_Position.xy = pos0.xy - offset;
  gl_Position.zw = pos0.zw;
  EmitVertex ();

  gl_Position.xy = pos0.xy + offset;
  EmitVertex ();

  offset         = pos1.w  * up;

  gl_Position.xy = pos1.xy - offset;
  gl_Position.zw = pos1.zw;
  EmitVertex ();

  gl_Position.xy = pos1.xy + offset;
  EmitVertex ();

  EndPrimitive();
}

)glsl";


//-----------------------------------------------------------------------
//   GEOM_LINES_MULTI_
//-----------------------------------------------------------------------


const char* ShaderSource::GEOM_LINES_MULTI_ = R"glsl(

#version 330 core

layout(lines)                           in;
layout(triangle_strip, max_vertices=4)  out;

uniform vec2   uniFrameSize;
uniform float  uniWidth;

in      vec4   geomColor[];

out     vec4   fragColor;


void main()
{
  vec4  pos0 = gl_in[0].gl_Position;
  vec4  pos1 = gl_in[1].gl_Position;
  vec2  dir  = normalize ( (pos1.xy / pos1.w) - (pos0.xy / pos0.w) );
  vec2  up   = uniWidth * vec2(  dir.y / uniFrameSize.x,
                                -dir.x / uniFrameSize.y );

  vec2  offset;


  fragColor      = geomColor[0];
  offset         = pos0.w  * up;

  gl_Position.xy = pos0.xy - offset;
  gl_Position.zw = pos0.zw;
  EmitVertex ();

  gl_Position.xy = pos0.xy + offset;
  EmitVertex ();

  fragColor      = geomColor[1];
  offset         = pos1.w  * up;

  gl_Position.xy = pos1.xy - offset;
  gl_Position.zw = pos1.zw;
  EmitVertex ();

  gl_Position.xy = pos1.xy + offset;
  EmitVertex ();

  EndPrimitive();
}

)glsl";


//-----------------------------------------------------------------------
//   GEOM_LINES_TEX1D_
//-----------------------------------------------------------------------


const char* ShaderSource::GEOM_LINES_TEX1D_ = R"glsl(

#version 330 core

layout(lines)                           in;
layout(triangle_strip, max_vertices=4)  out;

uniform vec2   uniFrameSize;
uniform float  uniWidth;

in      float  geomVxColor[];

out     float  U;


void main()
{
  vec4  pos0 = gl_in[0].gl_Position;
  vec4  pos1 = gl_in[1].gl_Position;
  vec2  dir  = normalize ( (pos1.xy / pos1.w) - (pos0.xy / pos0.w) );
  vec2  up   = uniWidth * vec2(  dir.y / uniFrameSize.x,
                                -dir.x / uniFrameSize.y );

  vec2  offset;


  U              = geomVxColor[0];
  offset         = pos0.w  * up;

  gl_Position.xy = pos0.xy - offset;
  gl_Position.zw = pos0.zw;
  EmitVertex ();

  gl_Position.xy = pos0.xy + offset;
  EmitVertex ();

  U              = geomVxColor[1];
  offset         = pos1.w  * up;

  gl_Position.xy = pos1.xy - offset;
  gl_Position.zw = pos1.zw;
  EmitVertex ();

  gl_Position.xy = pos1.xy + offset;
  EmitVertex ();

  EndPrimitive();
}

)glsl";


//-----------------------------------------------------------------------
//   GEOM_LINES_STIPPLE_CONST_
//-----------------------------------------------------------------------


const char* ShaderSource::GEOM_LINES_STIPPLE_CONST_ = R"glsl(

#version 330 core

layout(lines)                           in;
layout(triangle_strip, max_vertices=4)  out;

uniform vec2   uniFrameSize;
uniform float  uniWidth;

out     float  texCoord;


void main()
{
  vec4  pos0    = gl_in[0].gl_Position;
  vec4  pos1    = gl_in[1].gl_Position;

  vec2  winPos0 = pos0.xy / pos0.w;
  vec2  winPos1 = pos1.xy / pos1.w;
  vec2  dir     = winPos1 - winPos0;

  vec2  up;
  vec2  offset;


  up             = normalize ( dir );
  up             = uniWidth * vec2(  up.y / uniFrameSize.x,
                                    -up.x / uniFrameSize.y );

  texCoord       = 0.0F;
  offset         = pos0.w  * up;

  gl_Position.xy = pos0.xy - offset;
  gl_Position.zw = pos0.zw;
  EmitVertex ();

  gl_Position.xy = pos0.xy + offset;
  EmitVertex ();

  texCoord       = 0.5F * length( uniFrameSize * dir );
  offset         = pos1.w  * up;

  gl_Position.xy = pos1.xy - offset;
  gl_Position.zw = pos1.zw;
  EmitVertex ();

  gl_Position.xy = pos1.xy + offset;
  EmitVertex ();

  EndPrimitive();
}

)glsl";


//-----------------------------------------------------------------------
//   FRAG_LINES_STIPPLE_CONST_
//-----------------------------------------------------------------------


const char* ShaderSource::FRAG_LINES_STIPPLE_CONST_ = R"glsl(

#version 330 core

uniform vec4   uniColor;
uniform uint   uniFactor;
uniform uint   uniPattern;
uniform float  uniAlphaThreshold;

in      float  texCoord;

out     vec4   color;

void main()
{
  float  linePos = texCoord;
  uint   bit     = uint( round( linePos / uniFactor ) ) & 15U;

  // Alpha testing

  if ( uniColor.a < uniAlphaThreshold )
  {
    discard;
  }

  // Line pattern based on the bit pattern of uniPattern

  if ( (uniPattern & (1U << bit)) == 0U )
  {
    discard;
  }

  color = uniColor;
}

)glsl";


//=======================================================================
//   member functions
//=======================================================================

//-----------------------------------------------------------------------
//  load
//-----------------------------------------------------------------------


void ShaderSource::load

  ( GLContext&     gtx,
    ShaderProg&    shader,
    const String&  name )

{
  if      ( name == ShaderNames::FLAT_CONST_COLOR )
  {
    shader.compileSource ( gtx,
                           VERT_FLAT_CONST_COLOR_,
                           FRAG_FLAT_CONST_COLOR_ );
  }
  else if ( name == ShaderNames::FLAT_MULTI_COLOR )
  {
    shader.compileSource ( gtx,
                           VERT_FLAT_MULTI_COLOR_,
                           FRAG_FLAT_MULTI_COLOR_ );
  }
  else if ( name == ShaderNames::FLAT_TEX1D )
  {
    shader.compileSource ( gtx,
                           VERT_FLAT_TEX1D_,
                           FRAG_FLAT_TEX1D_ );
  }
  else if ( name == ShaderNames::FLAT_TEX1D_ALPHA )
  {
    shader.compileSource ( gtx,
                           VERT_FLAT_TEX1D_,
                           FRAG_FLAT_TEX1D_ALPHA_ );
  }
  else if ( name == ShaderNames::FLAT_TEX2D )
  {
    shader.compileSource ( gtx,
                           VERT_FLAT_TEX2D_,
                           FRAG_FLAT_TEX2D_ );
  }
  else if ( name == ShaderNames::FLAT_TEX2D_ALPHA )
  {
    shader.compileSource ( gtx,
                           VERT_FLAT_TEX2D_,
                           FRAG_FLAT_TEX2D_ALPHA_ );
  }
  else if ( name == ShaderNames::SHADE_CONST_COLOR )
  {
    shader.compileSource ( gtx,
                           VERT_SHADE_CONST_COLOR_,
                           FRAG_SHADE_CONST_COLOR_ );
  }
  else if ( name == ShaderNames::SHADE_MULTI_COLOR )
  {
    shader.compileSource ( gtx,
                           VERT_SHADE_MULTI_COLOR_,
                           FRAG_SHADE_MULTI_COLOR_ );
  }
  else if ( name == ShaderNames::SHADE_TEX1D )
  {
    shader.compileSource ( gtx,
                           VERT_SHADE_TEX1D_,
                           FRAG_SHADE_TEX1D_ );
  }
  else if ( name == ShaderNames::SHADE_TEX1D_ALPHA )
  {
    shader.compileSource ( gtx,
                           VERT_SHADE_TEX1D_,
                           FRAG_SHADE_TEX1D_ALPHA_ );
  }
  else if ( name == ShaderNames::SHADE_TEX2D )
  {
    shader.compileSource ( gtx,
                           VERT_SHADE_TEX2D_,
                           FRAG_SHADE_TEX2D_ );
  }
  else if ( name == ShaderNames::SHADE_TEX2D_ALPHA )
  {
    shader.compileSource ( gtx,
                           VERT_SHADE_TEX2D_,
                           FRAG_SHADE_TEX2D_ALPHA_ );
  }
  else if ( name == ShaderNames::ACCUMULATION )
  {
    shader.compileSource ( gtx,
                           VERT_CONST_ACCUMULATION_,
                           FRAG_CONST_ACCUMULATION_ );
  }
  else if ( name == ShaderNames::ORBS_UNIFORM )
  {
    shader.compileSource ( gtx,
                           VERT_UNIFORM_ORB_,
                           FRAG_UNIFORM_ORB_,
                           GEOM_UNIFORM_ORB_ );
  }
  else if ( name == ShaderNames::ORBS_VARYING )
  {
    shader.compileSource ( gtx,
                           VERT_VARYING_ORB_,
                           FRAG_VARYING_ORB_,
                           GEOM_VARYING_ORB_ );
  }
  else if ( name == ShaderNames::ORBS_CHECKER )
  {
    shader.compileSource ( gtx,
                           VERT_UNIFORM_ORB_,
                           FRAG_CHECKER_ORB_,
                           GEOM_CHECKER_ORB_ );
  }
  else if ( name == ShaderNames::LINES_CONST_COLOR )
  {
    shader.compileSource ( gtx,
                           VERT_FLAT_CONST_COLOR_,
                           FRAG_FLAT_CONST_COLOR_,
                           GEOM_LINES_CONST_ );
  }
  else if ( name == ShaderNames::LINES_MULTI_COLOR )
  {
    shader.compileSource ( gtx,
                           VERT_LINES_MULTI_,
                           FRAG_FLAT_MULTI_COLOR_,
                           GEOM_LINES_MULTI_ );
  }
  else if ( name == ShaderNames::LINES_TEX1D )
  {
    shader.compileSource ( gtx,
                           VERT_LINES_TEX1D_,
                           FRAG_FLAT_TEX1D_,
                           GEOM_LINES_TEX1D_ );
  }
  else if ( name == ShaderNames::LINES_STIPPLE )
  {
    shader.compileSource ( gtx,
                           VERT_FLAT_CONST_COLOR_,
                           FRAG_LINES_STIPPLE_CONST_,
                           GEOM_LINES_STIPPLE_CONST_ );
  }
  else
  {
    throw GLException (
      JEM_FUNC,
      String::format (
        "unable to compile shader program; unknown program name: %s",
        name
      )
    );
  }
}


//-----------------------------------------------------------------------
//  getProgram
//-----------------------------------------------------------------------


ShaderProg& ShaderSource::getProgram

  ( GLContext&     gtx,
    const String&  name )

{
  Ref<ShaderProg>  prog =

    dynamicCast<ShaderProg> ( gtx.findProgram( name ) );

  if ( ! prog )
  {
    prog = newInstance<ShaderProg> ( name );

    // Store the shader program in the GL context before compiling
    // the shader source so the shader program is only created once
    // in the case that an exception is thrown during the compilation
    // process.

    gtx.storeProgram ( name, prog );
    load ( gtx, *prog, name );
  }

  return *prog;
}


JEM_END_PACKAGE( gl )
