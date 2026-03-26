<?php

namespace Modules\Inspection\Http\Controllers;

use Illuminate\Routing\Controller;
use Modules\Inspection\Entities\InspectionTemplate;
use Modules\Inspection\Entities\InspectionTemplateItem;
use Modules\Inspection\Http\Requests\StoreInspectionTemplateItemRequest;

class InspectionTemplateItemController extends Controller
{
    public function store(StoreInspectionTemplateItemRequest $request, $templateId)
    {
        $template = InspectionTemplate::findOrFail($templateId);

        $template->items()->create($request->validated());

        return back()->with('success', __('inspection::messages.template_item_added'));
    }

    public function destroy($templateId, $itemId)
    {
        $template = InspectionTemplate::findOrFail($templateId);
        $item = $template->items()->where('id', $itemId)->firstOrFail();
        $item->delete();

        return back()->with('success', __('inspection::messages.template_item_removed'));
    }
}
