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

#include "IRenderer.h"
#include "Drawing.h"

namespace drawing
{
	Drawing::Drawing()
		: mRoot() {
		//
	}

	void Drawing::Render(IRenderer& target) {
		target.Begin_Render();
		mRoot.RenderTo(target);
		target.Finalize_Render();
	}

	Group& Drawing::Root() {
		return mRoot;
	}

	void Group::RenderContents(IRenderer& renderer) {
		for (auto& shapePtr : mObjects) {
			shapePtr->RenderTo(renderer);
		}
	}

	void Group::Apply_Defaults(Element& target) {
		try {
			drawing::Element& shapeTarget = dynamic_cast<drawing::Element&>(target);
			mDefault_Shape.CloneTo(shapeTarget);
		}
		catch (...) {
			// attempting to set default to an object, which is unexpected (not derived from vector shape)
		}
	}

	/* shape render methods */

	void Group::RenderTo(IRenderer& renderer) {
		renderer.Render_Group(*this);
	}

	void Circle::RenderTo(IRenderer& renderer) {
		renderer.Render_Circle(*this);
	}

	void Line::RenderTo(IRenderer& renderer) {
		renderer.Render_Line(*this);
	}

	void PolyLine::RenderTo(IRenderer& renderer) {
		renderer.Render_PolyLine(*this);
	}

	void Polygon::RenderTo(IRenderer& renderer) {
		renderer.Render_Polygon(*this);
	}

	void Rectangle::RenderTo(IRenderer& renderer) {
		renderer.Render_Rectangle(*this);
	}

	void Text::RenderTo(IRenderer& renderer) {
		renderer.Render_Text(*this);
	}
}
