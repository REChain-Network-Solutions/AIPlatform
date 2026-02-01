<?php

namespace App\Extensions\ProductPhotography\System\Http\Controllers;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\ProductPhotography\System\Http\Requests\ProductPhotographyRequest;
use App\Extensions\ProductPhotography\System\Models\UserPebblely;
use App\Extensions\ProductPhotography\System\Services\PebblelyService;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Support\Facades\Auth;

class ProductPhotographyController extends Controller
{
    public function __construct(
        public PebblelyService $service
    ) {}

    public function index(): View
    {
        return view('product-photography::index', [
            'last'   => UserPebblely::query()->where('user_id', Auth::id())->latest()->first(),
            'themes' => $this->service->getThemes(),
            'images' => UserPebblely::query()->where('user_id', Auth::id())->latest()->get(),
        ]);
    }

    public function store(ProductPhotographyRequest $request): JsonResponse|RedirectResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $request->validated();

        $driver = Entity::driver(EntityEnum::PEBBLELY)->inputImageCount(1)->calculateCredit();

        try {
            $driver->redirectIfNoCreditBalance();
        } catch (Exception $e) {
            return redirect()->back()->with([
                'message' => $e->getMessage(),
                'type'    => 'You have no credits left. Please consider upgrading your plan.',
            ]);
        }

        $removedImage = $this->service->removeBg($request->file('image'));

        $response = $this->service->createBg($removedImage, $request->get('background'));

        if (! isset($response['error'])) {

            UserPebblely::query()->create([
                'user_id' => Auth::id(),
                'image'   => $response,
            ]);

            $driver->decreaseCredit();

            return redirect()->back()->with([
                'message' => __('Image Created Successfully'),
                'type'    => 'success',
            ]);
        }

        return redirect()->back()->with([
            'message' => str_replace(["\r", "\n"], '', $response['message'])
                ?? __('UserPebblely API Key Error'),
            'type' => 'error',
        ]);
    }

    public function delete(string $id): JsonResponse|RedirectResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $model = UserPebblely::query()->findOrFail($id);

        $model->delete();

        return redirect()->back()->with(['message' => __('Deleted Successfully'), 'type' => 'success']);
    }
}
