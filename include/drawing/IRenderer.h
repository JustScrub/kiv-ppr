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

#include "Drawing.h"

namespace drawing {

	/* drawing renderer interface */
	class IRenderer {
		protected:
			double mCanvas_Width;
			double mCanvas_Height;

		public:
			IRenderer() : mCanvas_Width(0), mCanvas_Height(0) {}
			IRenderer(double canvasWidth, double canvasHeight) : mCanvas_Width(canvasWidth), mCanvas_Height(canvasHeight) {}

			// disallow copy and move, renderer is supposed to be one-shot instance
			IRenderer(const IRenderer&) = delete;
			IRenderer& operator=(const IRenderer&) = delete;
			IRenderer(IRenderer&&) = delete;
			IRenderer& operator=(IRenderer&&) = delete;

			double Get_Canvas_Width() const {
				return mCanvas_Width;
			}

			double Get_Canvas_Height() const {
				return mCanvas_Height;
			}

			void Set_Canvas_Width(double canvasWidth) {
				mCanvas_Width = canvasWidth;
			}

			void Set_Canvas_Height(double canvasHeight) {
				mCanvas_Height = canvasHeight;
			}

			void Set_Canvas_Size(double canvasWidth, double canvasHeight) {
				mCanvas_Width = canvasWidth;
				mCanvas_Height = canvasHeight;
			}

			virtual void Begin_Render() = 0;
			virtual void Finalize_Render() = 0;

			virtual void Render_Circle(drawing::Circle& shape) = 0;
			virtual void Render_Line(drawing::Line& shape) = 0;
			virtual void Render_PolyLine(drawing::PolyLine& shape) = 0;
			virtual void Render_Rectangle(drawing::Rectangle& shape) = 0;
			virtual void Render_Polygon(drawing::Polygon& shape) = 0;
			virtual void Render_Text(drawing::Text& shape) = 0;
			virtual void Render_Group(drawing::Group& shape) = 0;
	};
}
