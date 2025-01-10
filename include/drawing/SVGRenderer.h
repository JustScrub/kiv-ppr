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

#include "IRenderer.h"
#include "Drawing.h"

#include <string>
#include <sstream>

/* renderer implementation for SVG file format */
class CSVG_Renderer : public drawing::IRenderer {
	private:
		std::string &mSvg_String_Target;
		std::ostringstream mSvg_Target;

		size_t mRenderer_Depth = 0;

	protected:
		static std::string Color_To_String(const RGBColor& color);
		static std::string Anchor_To_String(const drawing::Text::TextAnchor anchor);
		static std::string Weight_To_String(const drawing::Text::FontWeight weight);
		void Render_Default_Params(drawing::Element& shape);

	public:
		CSVG_Renderer(double canvasWidth, double canvasHeight, std::string& svgTarget);

		virtual void Begin_Render() override;
		virtual void Finalize_Render() override;

		virtual void Render_Circle(drawing::Circle& shape) override;
		virtual void Render_Line(drawing::Line& shape) override;
		virtual void Render_PolyLine(drawing::PolyLine& shape) override;
		virtual void Render_Rectangle(drawing::Rectangle& shape) override;
		virtual void Render_Polygon(drawing::Polygon& shape) override;
		virtual void Render_Text(drawing::Text& shape) override;
		virtual void Render_Group(drawing::Group& shape) override;
};
