#ifndef NYUPARSER_H
#define NYUPARSER_H

#include "parse/tokens.h"
#include "Scene.h"
#include "geom/Transformation.h"
#include "structs/vector.h"

class NYUParser{
   private:
      Tokenizer * tokenizer;

      // helper functions
      void ParseLeftAngle();
      void ParseRightAngle();
      double ParseDouble();
      void ParseLeftCurly();
      void ParseRightCurly();
      void ParseComma();
      void ParseVector(vec3_t & v);
      void ParseRGBFColor(color_t & c, float & f);
      void ParseRGBColor(color_t & c, float & f);
      void ParseColor(color_t & c, float & f);
      void PrintColor(color_t & c, float & f);
      void ParseMatrix();
      void ParseTransform(Geometry & s);
      void Parsepigment_t(pigment_t & p);
      void Printpigment_t(pigment_t & p);
      void Parsefinish_t(finish_t & f);
      void Printfinish_t(finish_t & f);
      // void ParseInterior( struct Interior* interior);
      void ParseInterior(float & ior);
      // void InitModifiers(struct ModifierStruct* modifiers);
      // void ParseModifiers(struct ModifierStruct* modifiers);
      // void PrintModifiers(struct ModifierStruct* modifiers);
      void ParseModifiers(Geometry & s);
      void PrintModifiers(Geometry & s);
      // void ParseCamera();
      Camera * ParseCamera();
      void ParsePolygon();
      // void ParseSphere();
      Sphere * ParseSphere();
      // void ParseBox();
      Box * ParseBox();
      void ParseCylinder();
      // void ParseCone();
      // Cone * ParseCone();
      void ParseQuadric();
      Triangle * ParseTriangle(); // added by mbecker
      Plane * ParsePlane(); // added by mbecker
      // void ParseLightSource();
      Light * ParseLightSource();
      void ParseGlobalSettings();

   public:
      void parse(std::fstream & input, Scene & s);
      // Parse(FILE* infile);
};

#endif
