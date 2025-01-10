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

#include "SVGRenderer.h"

CSVG_Renderer::CSVG_Renderer(double canvasWidth, double canvasHeight, std::string& svgTarget)
	: drawing::IRenderer(canvasWidth, canvasHeight), mSvg_String_Target(svgTarget) {
	//
}

std::string CSVG_Renderer::Color_To_String(const RGBColor& color) {
	return "rgb(" + std::to_string(color.r) + "," + std::to_string(color.g) + "," + std::to_string(color.b) + ")";
}

std::string CSVG_Renderer::Anchor_To_String(const drawing::Text::TextAnchor anchor) {
	switch (anchor) {
		case drawing::Text::TextAnchor::START:
			return "start";
		case drawing::Text::TextAnchor::MIDDLE:
			return "middle";
		case drawing::Text::TextAnchor::END:
			return "end";
	}

	return "";
}

std::string CSVG_Renderer::Weight_To_String(const drawing::Text::FontWeight weight) {
	switch (weight) {
		case drawing::Text::FontWeight::LIGHT:
			return "light";
		case drawing::Text::FontWeight::NORMAL:
			return "normal";
		case drawing::Text::FontWeight::BOLD:
			return "bold";
	}

	return "normal";
}

void CSVG_Renderer::Render_Default_Params(drawing::Element& shape) {
	if (!shape.Get_Id().empty()) {
		mSvg_Target << " id=\"" << shape.Get_Id() << "\"";
	}
	if (!shape.Get_Class().empty()) {
		mSvg_Target << " class=\"" << shape.Get_Class() << "\"";
	}
	if (!shape.Get_Transform().empty()) {
		mSvg_Target << " transform=\"" << shape.Get_Transform() << "\"";
	}

	if (shape.Get_Stroke_Width() > 0) {
		mSvg_Target << " stroke-width=\"" << shape.Get_Stroke_Width() << "\"";
		mSvg_Target << " stroke=\"" << Color_To_String(shape.Get_Stroke_Color()) << "\"";
		mSvg_Target << " stroke-linejoin=\"round\""; // TODO: line join parameter
		mSvg_Target << " stroke-linecap=\"round\""; // TODO: line cap parameter
		mSvg_Target << " stroke-opacity=\"" << shape.Get_Stroke_Opacity() << "\"";
	}

	const auto& dashArray = shape.Get_Stroke_Dash_Array();
	if (!dashArray.empty()) {
		mSvg_Target << " stroke-dasharray=\"";
		for (size_t i = 0; i < dashArray.size(); i++) {
			mSvg_Target << dashArray[i] << " ";
		}
		mSvg_Target << "\"";
	}

	mSvg_Target << " fill=\"" << Color_To_String(shape.Get_Fill_Color()) << "\"";
	mSvg_Target << " fill-opacity=\"" << shape.Get_Fill_Opacity() << "\"";
}

void CSVG_Renderer::Begin_Render() {
	if (mRenderer_Depth == 0) {
		mSvg_Target.clear();
		mSvg_Target << "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\""
					<< " viewBox=\"0 0 " << Get_Canvas_Width() << " " << Get_Canvas_Height() << "\""
					<< " id=\"svgContainer\" "
					<< " shape-rendering=\"geometricPrecision\">"
					<< std::endl;

		mSvg_Target << "<g>" << std::endl;
	}

	mRenderer_Depth++;
}

void CSVG_Renderer::Finalize_Render() {
	mRenderer_Depth--;

	if (mRenderer_Depth == 0) {
		mSvg_Target << "</g>" << std::endl;
		mSvg_Target << "</svg>" << std::endl;

		mSvg_String_Target = mSvg_Target.str();
		mSvg_Target.clear();
	}
}

void CSVG_Renderer::Render_Circle(drawing::Circle& shape) {
	mSvg_Target << "<circle cx=\"" << shape.Get_Position_X() << "\" cy=\"" << shape.Get_Position_Y() << "\" r=\"" << shape.Get_Radius() << "\"";
	Render_Default_Params(shape);
	mSvg_Target << " />" << std::endl;
}

void CSVG_Renderer::Render_Line(drawing::Line& shape) {
	mSvg_Target << "<line x1=\"" << shape.Get_Position_X() << "\" y1=\"" << shape.Get_Position_Y() << "\" x2=\"" << shape.Get_Target_X() << "\" y2=\"" << shape.Get_Target_Y() << "\"";
	Render_Default_Params(shape);
	mSvg_Target << " />" << std::endl;
}

void CSVG_Renderer::Render_PolyLine(drawing::PolyLine& shape) {
	mSvg_Target << "<polyline points=\"";

	mSvg_Target << shape.Get_Position_X() << "," << shape.Get_Position_Y();
	for (auto& pt : shape.Get_Points()) {
		mSvg_Target << " " << pt.x << "," << pt.y;
	}
	mSvg_Target << "\"" << std::endl;

	Render_Default_Params(shape);
	mSvg_Target << " />" << std::endl;
}

void CSVG_Renderer::Render_Rectangle(drawing::Rectangle& shape) {
	mSvg_Target << "<rect x=\"" << shape.Get_Position_X() << "\" y=\"" << shape.Get_Position_Y() << "\" width=\"" << shape.Get_Width() << "\" height=\"" << shape.Get_Height() << "\"";
	Render_Default_Params(shape);
	mSvg_Target << "/>" << std::endl;
}

void CSVG_Renderer::Render_Polygon(drawing::Polygon& shape) {
	mSvg_Target << "<polygon points=\"";
	mSvg_Target << shape.Get_Position_X() << "," << shape.Get_Position_Y();
	for (auto& pt : shape.Get_Points()) {
		mSvg_Target << " " << pt.x << "," << pt.y;
	}
	mSvg_Target << "\"" << std::endl;

	Render_Default_Params(shape);
	mSvg_Target << " />" << std::endl;
}

void CSVG_Renderer::Render_Text(drawing::Text& shape) {
	mSvg_Target << "<text x=\"" << shape.Get_Position_X() << "\" y=\"" << shape.Get_Position_Y() << "\""
				<< " text-anchor=\"" << Anchor_To_String(shape.Get_Anchor()) << "\"";

	Render_Default_Params(shape);

	mSvg_Target << " font-weight=\"" << Weight_To_String(shape.Get_Font_Weight()) << "\" font-size=\"" << shape.Get_Font_Size() << "\">";
	mSvg_Target << shape.Get_Text();
	mSvg_Target << "</text>" << std::endl;
}

void CSVG_Renderer::Render_Group(drawing::Group& shape) {
	mSvg_Target << "<g";
	if (shape.Get_Add_Stroke()) {
		Render_Default_Params(shape);
	}
	mSvg_Target << ">" << std::endl;

	shape.RenderContents(*this);

	mSvg_Target << "</g>" << std::endl;
}
