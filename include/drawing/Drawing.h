/**
 * SmartCGMS - continuous glucose monitoring and controlling framework
 * https://diabetes.zcu.cz/
 *
 * Copyright (c) since 2018 University of West Bohemia.
 *
 * Contact:
 * diabetes@mail.kiv.zcu.cz
 * Medical Informatics, Department of Computer Science and Engineering
 * Faculty of Applied Sciences, University of West Bohemia
 * Univerzitni 8, 301 00 Pilsen
 * Czech Republic
 * 
 * 
 * Purpose of this software:
 * This software is intended to demonstrate work of the diabetes.zcu.cz research
 * group to other scientists, to complement our published papers. It is strictly
 * prohibited to use this software for diagnosis or treatment of any medical condition,
 * without obtaining all required approvals from respective regulatory bodies.
 *
 * Especially, a diabetic patient is warned that unauthorized use of this software
 * may result into severe injure, including death.
 *
 *
 * Licensing terms:
 * Unless required by applicable law or agreed to in writing, software
 * distributed under these license terms is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * a) This file is available under the Apache License, Version 2.0.
 * b) When publishing any derivative work or results obtained using this software, you agree to cite the following paper:
 *    Tomas Koutny and Martin Ubl, "SmartCGMS as a Testbed for a Blood-Glucose Level Prediction and/or 
 *    Control Challenge with (an FDA-Accepted) Diabetic Patient Simulation", Procedia Computer Science,  
 *    Volume 177, pp. 354-362, 2020
 */

#pragma once

#include <vector>
#include <memory>
#include <string>

#include "IRenderer.h"

/* a class representing RGB-encoded color */
struct RGBColor {

	uint8_t r;
	uint8_t g;
	uint8_t b;

	uint8_t a = 255;

	/* converts the hex character to an integer */
	static constexpr uint8_t Hex_Lookup(char c) {
		if (c >= '0' && c <= '9') {
			return c - '0';
		}
		else if (c >= 'A' || c <= 'F') {
			return 10 + (c - 'A');
		}
		else if (c >= 'a' || c <= 'f') {
			return 10 + (c - 'a');
		}
		return 0;
	}

	/* converts HTML-encoded color (e.g., #FF0055) into a RGBColor instance */
	static RGBColor From_HTML_Color(const std::string& col) {

		if (col.length() == 9) { // in case of "#AARRGGBB" format (used in our color descriptors)
			return {
				static_cast<uint8_t>(Hex_Lookup(col[3]) * 16 + Hex_Lookup(col[4])),
				static_cast<uint8_t>(Hex_Lookup(col[5]) * 16 + Hex_Lookup(col[6])),
				static_cast<uint8_t>(Hex_Lookup(col[7]) * 16 + Hex_Lookup(col[8])),
				static_cast<uint8_t>(Hex_Lookup(col[1]) * 16 + Hex_Lookup(col[2]))
			};
		}

		if (col.length() != 7) { // otherwise we expect "#RRGGBB" format
			return { 0,0,0,255 };
		}

		return {
			static_cast<uint8_t>(Hex_Lookup(col[1]) * 16 + Hex_Lookup(col[2])),
			static_cast<uint8_t>(Hex_Lookup(col[3]) * 16 + Hex_Lookup(col[4])),
			static_cast<uint8_t>(Hex_Lookup(col[5]) * 16 + Hex_Lookup(col[6])),
			255
		};
	}

	/* converts int-encoded color to RGBColor instance */
	static RGBColor From_UInt32(const uint32_t col, bool has_alpha = false) {
		return {
			static_cast<uint8_t>(col >> 16),
			static_cast<uint8_t>(col >> 8),
			static_cast<uint8_t>(col),
			has_alpha ? static_cast<uint8_t>(col >> 24) : static_cast<uint8_t>(255),
		};
	}
};

namespace drawing {

	class IRenderer;

	/* drawing point structure */
	struct Point {
		Point(double _x, double _y) : x(_x), y(_y) { }

		double x;
		double y;
	};

	/* drawable canvas element */
	class Element {
		protected:
			double mPosition_X = 0.0;
			double mPosition_Y = 0.0;

			double mStroke_Width{ 0.0 };
			RGBColor mStroke_Color{ 0, 0, 0 };
			RGBColor mFill_Color{ 0, 0, 0 };
			double mStroke_Opacity{ 1.0 }; // 0.0 (not visible) - 1.0 (max opaque)
			double mFill_Opacity{ 1.0 }; // 0.0 (not visible) - 1.0 (max opaque)
			std::string mId{ };
			std::string mClass{ };
			std::vector<double> mStroke_Dash_Array{ };
			std::string mTransform{ };

		public:
			Element() = default;
			explicit Element(double x, double y) {
				Set_Position(x, y);
			}

			virtual ~Element() = default;

			Element(const Element& other) {
				Set_Position(other.mPosition_X, other.mPosition_Y);
			}

			Element& operator=(const Element& other) {
				Set_Position(other.mPosition_X, other.mPosition_Y);
				return *this;
			}

			double Get_Position_X() const {
				return mPosition_X;
			}

			double Get_Position_Y() const {
				return mPosition_Y;
			}

			void Set_Position_X(double x) {
				mPosition_X = x;
			}

			void Set_Position_Y(double y) {
				mPosition_Y = y;
			}

			void Set_Position(double x, double y) {
				mPosition_X = x;
				mPosition_Y = y;
			}

			double Get_Stroke_Width() const {
				return mStroke_Width;
			}

			RGBColor Get_Stroke_Color() const {
				return mStroke_Color;
			}

			RGBColor Get_Fill_Color() const {
				return mFill_Color;
			}

			double Get_Stroke_Opacity() const {
				return mStroke_Opacity;
			}

			double Get_Fill_Opacity() const {
				return mFill_Opacity;
			}

			const std::string& Get_Id() const {
				return mId;
			}

			const std::string& Get_Class() const {
				return mClass;
			}

			const std::vector<double>& Get_Stroke_Dash_Array() {
				return mStroke_Dash_Array;
			}

			const std::string& Get_Transform() const {
				return mTransform;
			}

			Element& Set_Stroke_Width(double width) {
				mStroke_Width = width;
				return *this;
			}

			Element& Set_Stroke_Color(RGBColor color) {
				mStroke_Color = color;
				return *this;
			}

			Element& Set_Fill_Color(RGBColor color) {
				mFill_Color = color;
				return *this;
			}

			Element& Set_Stroke_Opacity(double opacity) {
				mStroke_Opacity = opacity;
				return *this;
			}

			Element& Set_Fill_Opacity(double opacity) {
				mFill_Opacity = opacity;
				return *this;
			}

			Element& Set_Id(const std::string& id) {
				mId = id;
				return *this;
			}

			Element& Set_Class(const std::string& clss) {
				mClass = clss;
				return *this;
			}

			Element& Set_Stroke_Dash_Array(const std::vector<double>& dashArray) {
				mStroke_Dash_Array = dashArray;
				return *this;
			}

			Element& Set_Transform(const std::string& transform) {
				mTransform = transform;
				return *this;
			}

			// clone settings to another Shape
			void CloneTo(Element& target) const {
				target.mStroke_Width = mStroke_Width;
				target.mStroke_Color = mStroke_Color;
				target.mFill_Color = mFill_Color;
				target.mStroke_Opacity = mStroke_Opacity;
				target.mFill_Opacity = mFill_Opacity;
				// do not copy ID - by design ID is unique
				target.mClass = mClass;
				target.mStroke_Dash_Array = mStroke_Dash_Array;
			}

			virtual void RenderTo(IRenderer& renderer) {
			};
	};

	using UElement = std::unique_ptr<drawing::Element>;

	/* drawable element group */
	class Group : public Element {
		protected:
			drawing::Element mDefault_Shape;
			std::vector<UElement> mObjects;

		private:
			bool mAddStroke = false;

		public:
			Group(const std::string& id = "")
				: Element(0, 0) {
				Set_Id(id);
			}

			bool Get_Add_Stroke() const {
				return mAddStroke;
			}

			Group& Set_Add_Stroke(bool add) {
				mAddStroke = add;
				return *this;
			}

			void RenderContents(IRenderer& renderer);
			virtual void RenderTo(IRenderer& renderer) override;

			template<typename T>
			T& Add() {
				mObjects.push_back(std::make_unique<T>());
				T& obj = static_cast<T&>(*(*mObjects.rbegin()).get());
				Apply_Defaults(obj);
				return obj;
			}

			template<typename T, typename... Args>
			T& Add(Args... args) {
				mObjects.push_back(std::make_unique<T>(std::forward<Args>(args)...));
				T& obj = static_cast<T&>(*(*mObjects.rbegin()).get());
				Apply_Defaults(obj);
				return obj;
			}

			virtual void Apply_Defaults(drawing::Element& target);

			double Get_Default_Stroke_Width() const {
				return mDefault_Shape.Get_Stroke_Width();
			}

			RGBColor Get_Default_Stroke_Color() const {
				return mDefault_Shape.Get_Stroke_Color();
			}

			RGBColor Get_Default_Fill_Color() const {
				return mDefault_Shape.Get_Fill_Color();
			}

			double Get_Default_Stroke_Opacity() const {
				return mDefault_Shape.Get_Stroke_Opacity();
			}

			double Get_Default_Fill_Opacity() const {
				return mDefault_Shape.Get_Fill_Opacity();
			}

			void Set_Default_Stroke_Width(double width) {
				mDefault_Shape.Set_Stroke_Width(width);
			}

			void Set_Default_Stroke_Color(RGBColor color) {
				mDefault_Shape.Set_Stroke_Color(color);
			}

			void Set_Default_Fill_Color(RGBColor color) {
				mDefault_Shape.Set_Fill_Color(color);
			}

			void Set_Default_Stroke_Opacity(double opacity) {
				mDefault_Shape.Set_Stroke_Opacity(opacity);
			}

			void Set_Default_Fill_Opacity(double opacity) {
				mDefault_Shape.Set_Fill_Opacity(opacity);
			}
	};

	/* drawable circle element */
	class Circle : public Element {

		protected:
			double mRadius;

		public:
			Circle(double posX = 0.0, double posY = 0.0, double radius = 0.0)
				: Element(posX, posY), mRadius(radius) {
			}

			double Get_Radius() const {
				return mRadius;
			}

			Circle& Set_Radius(double radius) {
				mRadius = radius;
				return *this;
			}

			virtual void RenderTo(IRenderer& renderer) override;
	};

	/* drawable line element */
	class Line : public Element {

		protected:
			double mTarget_X;
			double mTarget_Y;

		public:
			Line(double posX = 0.0, double posY = 0.0, double targetX = 0.0, double targetY = 0.0)
				: Element(posX, posY), mTarget_X(targetX), mTarget_Y(targetY) {
			}

			double Get_Target_X() const {
				return mTarget_X;
			}

			double Get_Target_Y() const {
				return mTarget_Y;
			}

			Line& Set_Target_X(double targetX) {
				mTarget_X = targetX;
				return *this;
			}

			Line& Set_Target_Y(double targetY) {
				mTarget_Y = targetY;
				return *this;
			}

			virtual void RenderTo(IRenderer& renderer) override;
	};

	/* drawable polyline element */
	class PolyLine : public Element {
		protected:
			std::vector<Point> mPoints;

		public:
			PolyLine(double posX = 0.0, double posY = 0.0)
				: Element(posX, posY) {
			}

			const std::vector<Point>& Get_Points() const {
				return mPoints;
			};

			PolyLine& Add_Point(double x, double y) {
				mPoints.emplace_back(x, y);
				return *this;
			}

			PolyLine& Clear() {
				mPoints.clear();
				return *this;
			}

			virtual void RenderTo(IRenderer& renderer) override;
	};

	/* drawable polygon element */
	class Polygon : public Element {
		protected:
			std::vector<Point> mPoints;

		public:
			Polygon(double posX = 0.0, double posY = 0.0)
				: Element(posX, posY) {
			}

			const std::vector<Point>& Get_Points() const {
				return mPoints;
			}

			Polygon& Add_Point(double x, double y) {
				mPoints.emplace_back(x, y);
				return *this;
			}

			Polygon& Clear() {
				mPoints.clear();
				return *this;
			}

			virtual void RenderTo(IRenderer& renderer) override;
	};

	/* drawable rectangle element */
	class Rectangle : public Element {
		protected:
			double mWidth;
			double mHeight;

		public:
			Rectangle(double posX = 0.0, double posY = 0.0, double width = 0.0, double height = 0.0)
				: Element(posX, posY), mWidth(width), mHeight(height) {
			}

			double Get_Width() const {
				return mWidth;
			}

			double Get_Height() const {
				return mHeight;
			}

			Rectangle& Set_Width(double width) {
				mWidth = width;
				return *this;
			}

			Rectangle& Set_Height(double height) {
				mHeight = height;
				return *this;
			}

			virtual void RenderTo(IRenderer& renderer) override;
	};

	class Text : public Element {
		public:
			enum class TextAnchor {
				START,
				MIDDLE,
				END
			};

			enum class FontWeight {
				LIGHT,
				NORMAL,
				BOLD
			};

		protected:
			std::string mText;
			TextAnchor mText_Anchor = TextAnchor::START;
			FontWeight mFont_Weight = FontWeight::NORMAL;
			double mFont_Size = 10.0;

		public:
			Text(double posX = 0.0, double posY = 0.0, const std::string& text = "")
				: Element(posX, posY), mText(text) {
			}

			const std::string& Get_Text() const {
				return mText;
			}

			TextAnchor Get_Anchor() const {
				return mText_Anchor;
			}

			FontWeight Get_Font_Weight() const {
				return mFont_Weight;
			}

			double Get_Font_Size() const {
				return mFont_Size;
			}

			Text& Set_Text(const std::string& text) {
				mText = text;
				return *this;
			}

			Text& Set_Anchor(TextAnchor anchor) {
				mText_Anchor = anchor;
				return *this;
			}

			Text& Set_Font_Weight(FontWeight weight) {
				mFont_Weight = weight;
				return *this;
			}

			Text& Set_Font_Size(double size) {
				mFont_Size = size;
				return *this;
			}

			virtual void RenderTo(IRenderer& renderer) override;
	};

	/* drawing container */
	class Drawing {
		protected:
			Group mRoot;

		public:
			Drawing();

			/* renders the contained element into the target renderer */
			virtual void Render(IRenderer& target);
			/* retrieves root element */
			Group& Root();
	};
}
