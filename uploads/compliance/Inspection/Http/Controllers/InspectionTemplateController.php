<?php

namespace Modules\Inspection\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Modules\Inspection\Entities\InspectionTemplate;
use Modules\Inspection\Http\Requests\StoreInspectionTemplateRequest;
use Modules\Inspection\Http\Requests\UpdateInspectionTemplateRequest;

class InspectionTemplateController extends Controller
{
    public function index()
    {
        $templates = InspectionTemplate::query()
            ->orderByDesc('id')
            ->paginate(20);

        return view('inspection::templates.index', compact('templates'));
    }

    public function create()
    {
        return view('inspection::templates.create');
    }

    public function store(StoreInspectionTemplateRequest $request)
    {
        $template = InspectionTemplate::create($request->validated());

        return redirect()
            ->route('inspection-templates.edit', $template->id)
            ->with('success', __('inspection::messages.template_created'));
    }

    public function show($id)
    {
        $template = InspectionTemplate::with('items')->findOrFail($id);

        return view('inspection::templates.show', compact('template'));
    }

    public function edit($id)
    {
        $template = InspectionTemplate::with('items')->findOrFail($id);

        return view('inspection::templates.edit', compact('template'));
    }

    public function update(UpdateInspectionTemplateRequest $request, $id)
    {
        $template = InspectionTemplate::findOrFail($id);
        $template->update($request->validated());

        return back()->with('success', __('inspection::messages.template_updated'));
    }

    public function destroy($id)
    {
        $template = InspectionTemplate::findOrFail($id);
        $template->delete();

        return redirect()
            ->route('inspection-templates.index')
            ->with('success', __('inspection::messages.template_deleted'));
    }
}
