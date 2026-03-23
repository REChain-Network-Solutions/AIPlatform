<?php

namespace App\Http\Controllers;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Entity\Models\Entity;
use App\Http\Requests\Admin\Chatbot\UpdateEngineImagesRequest;
use App\Models\Finance\AiChatModelPlan;
use App\Models\Plan;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Http\UploadedFile;

class AiChatbotModelController extends Controller
{
    public function index()
    {
        $enablesEngines = EngineEnum::whereHasEnabledModels();
        $plans = Plan::query()
            ->where('type', 'subscription')
            ->get();

        return view('panel.admin.chatbot.ai-models', compact('enablesEngines', 'plans'));
    }

    public function update(Request $request)
    {
        $data = $request->validate([
            'selected_title.*' => 'required',
            'selected_plans.*' => 'sometimes',
            'no_plan_users.*'  => 'sometimes',
        ]);

        foreach ($data['selected_title'] as $key => $value) {
            Entity::query()
                ->where('id', $key)
                ->update([
                    'selected_title' => $value,
                ]);
        }

        AiChatModelPlan::query()->delete();

        $selected_plans = $request->input('selected_plans');

        if ($selected_plans) {
            foreach ($selected_plans as $id => $value) {
                foreach ($value as $item) {
                    AiChatModelPlan::query()
                        ->create([
                            'plan_id'     => $item,
                            'entity_id'   => $id,
                        ]);
                }
            }
        }

        Entity::query()->update([
            'is_selected' => false,
        ]);

        $no_plan_users = $request->input('no_plan_users');

        if ($no_plan_users) {
            foreach ($no_plan_users as $key => $value) {
                Entity::query()
                    ->where('id', $key)
                    ->update([
                        'is_selected' => true,
                    ]);
            }
        }

        return redirect()->back()->with([
            'message' => 'AI Models updated successfully',
            'type'    => 'success',
        ]);
    }

    public function modelsIndex(): \Illuminate\Contracts\View\View
    {
        $engines = Entity::query()
            ->select('engine', 'image')
            ->get()
            ->groupBy('engine')
            ->map(fn ($entities, $engine) => (object) [
                'engine'       => $engine,
                'first_image'  => $entities->pluck('image')->filter()->first(),
                'entity_count' => $entities->count(),
            ])
            ->values();

        return view('panel.admin.chatbot.all-engines', compact('engines'));
    }

    public function updateEngineImages(UpdateEngineImagesRequest $request): RedirectResponse
    {
        $files = $request->file('engine_logo', []);
        $uploadDirRelative = 'upload/enginelogo';
        $uploadDirAbsolute = public_path($uploadDirRelative);

        if (! is_dir($uploadDirAbsolute)) {
            @mkdir($uploadDirAbsolute, 0755, true);
        }

        foreach ($files as $engineKey => $file) {
            if (! ($file instanceof UploadedFile)) {
                continue;
            }

            if (! $file->isValid()) {
                continue;
            }

            $safeEngineKey = preg_replace('/[^a-zA-Z0-9_\-]/', '_', (string) $engineKey);
            $extension = $file->getClientOriginalExtension() ?: 'png';
            $fileName = $safeEngineKey . '_logo_' . time() . '.' . $extension;

            $file->move($uploadDirAbsolute, $fileName);

            $relativePath = $uploadDirRelative . '/' . $fileName;

            // Update all entities for this engine with the new logo.
            // If no file is uploaded for an engine, we keep the old image (or null) untouched.
            Entity::query()
                ->where('engine', $engineKey)
                ->update(['image' => $relativePath]);
        }

        return redirect()->back()->with([
            'message' => 'Engine images updated successfully',
            'type'    => 'success',
        ]);
    }
}
