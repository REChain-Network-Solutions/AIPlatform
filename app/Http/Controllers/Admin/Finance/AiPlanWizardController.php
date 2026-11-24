<?php

namespace App\Http\Controllers\Admin\Finance;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Http\Middleware\Custom\FinanceLicenseMiddleware;
use App\Http\Requests\Admin\Finance\AiPlanWizard\ChatRequest;
use App\Http\Requests\Admin\Finance\AiPlanWizard\StorePlanRequest;
use App\Http\Requests\Admin\Finance\AiPlanWizard\ValidateStepRequest;
use App\Services\Finance\AiPlanWizardService;
use Illuminate\Http\JsonResponse;
use Symfony\Component\HttpFoundation\StreamedResponse;

class AiPlanWizardController extends Controller
{
    public function __construct(protected AiPlanWizardService $wizardService)
    {
        $this->middleware(FinanceLicenseMiddleware::class);
    }

    public function presets(): JsonResponse
    {
        return response()->json([
            'presets' => $this->wizardService->suggestPresets(),
        ]);
    }

    public function validateStep(ValidateStepRequest $request): JsonResponse
    {
        return response()->json(['ok' => true]);
    }

    public function chat(ChatRequest $request): StreamedResponse|JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json(['message' => __('This feature is disabled in demo mode.')], 403);
        }

        return $this->wizardService->streamChat(
            $request->validated('messages'),
            (array) $request->validated('draft', [])
        );
    }

    public function store(StorePlanRequest $request): JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json(['message' => __('This feature is disabled in demo mode.')], 403);
        }

        $plan = $this->wizardService->store($request->validated());

        return response()->json([
            'message'  => __(':name plan created successfully.', ['name' => $plan->name]),
            'edit_url' => $plan->type === 'prepaid'
                ? route('dashboard.admin.finance.token-pack-plan.edit', $plan->id)
                : route('dashboard.admin.finance.plan.edit', $plan->id),
        ]);
    }
}
