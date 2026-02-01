<?php

namespace App\Extensions\AIRealtimeImage\System\Http\Controllers;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\AIRealtimeImage\System\Enums\Status;
use App\Extensions\AIRealtimeImage\System\Http\Requests\AIRealtimeImageRequest;
use App\Extensions\AIRealtimeImage\System\Http\Resources\RealtimeImageResource;
use App\Extensions\AIRealtimeImage\System\Models\RealtimeImage;
use App\Extensions\AIRealtimeImage\System\Services\TogetherService;
use App\Http\Controllers\Controller;
use App\Models\Usage;
use Exception;

class AIRealtimeImageController extends Controller
{
    public function __construct(
        public TogetherService $togetherService
    ) {}

    public function index()
    {
        return view('ai-realtime-image::index', [
            'images' => RealtimeImage::query()
                ->where('status', Status::success->value)
                ->orderBy('created_at', 'desc')
                ->paginate(15),
        ]);
    }

    public function store(AIRealtimeImageRequest $request)
    {
        $data = $request->validated();

        $driver = Entity::driver(EntityEnum::BLACK_FOREST_LABS_FLUX_1_SCHNELL)->inputImageCount(1)->calculateCredit();

        try {
            $driver->redirectIfNoCreditBalance();
        } catch (Exception $e) {
            return response()->json([
                'message' => __('You have no credits left. Please consider upgrading your plan.'),
                'status'  => 'error',
                'type'    => 'error',
            ]);
        }

        $realtimeImage = RealtimeImage::query()->create($data);

        $realtimeImage = $this->togetherService->generate($realtimeImage);

        if ($realtimeImage->status === Status::success) {

            Usage::getSingle()->updateImageCounts($driver->calculate());

            $driver->decreaseCredit();

            return RealtimeImageResource::make($realtimeImage)->additional([
                'payload'        => $realtimeImage->payload,
                'formatted_date' => $realtimeImage->created_at->diffInMinutes() < 1 ? trans('Just now') : $realtimeImage->created_at->diffForHumans(),
                'status'         => $realtimeImage->status,
                'message'        => trans('Image generated successfully'),
            ]);
        }

        if ($request->user()->isAdmin()) {
            $response = $realtimeImage->getAttribute('response');

            $message = data_get($response, 'error.message');

            if ($message) {
                return response()->json([
                    'message' => $message,
                ], 422);
            }
        }

        return response()->json([
            'message' => trans('Failed to generate image'),
        ], 422);
    }

    public function gallery()
    {

        return view('ai-realtime-image::gallery.gallery', [
            'images' => RealtimeImage::query()
                ->where('status', Status::success->value)
                ->orderBy('created_at', 'desc')
                ->paginate(15),
        ]);
    }

    public function destroy() {}
}
