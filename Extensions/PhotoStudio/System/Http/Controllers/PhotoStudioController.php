<?php

namespace App\Extensions\PhotoStudio\System\Http\Controllers;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\PhotoStudio\System\Http\Requests\PhotoStudioRequest;
use App\Extensions\PhotoStudio\System\Models\PhotoStudio;
use App\Extensions\PhotoStudio\System\Services\NovitaService;
use App\Extensions\PhotoStudio\System\Services\PhotoStudioService;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Support\Facades\Auth;

class PhotoStudioController extends Controller
{
    public function __construct(
        public PhotoStudioService $service,
        public NovitaService $novitaService
    ) {}

    public function index(): View
    {
        $inProgress = PhotoStudio::query()->where('status', 'in_progress')->pluck('id')->toArray();

        return view('photo-studio::index', [
            'last'       => PhotoStudio::query()->where('user_id', auth()->id())->whereNot('status', 'in_progress')->latest()->first(),
            'images'     => PhotoStudio::query()->where('user_id', auth()->id())->latest()->get(),
            'inProgress' => $inProgress,
        ]);
    }

    public function store(PhotoStudioRequest $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $defaultPsEngine = setting('default_photo_studio', EntityEnum::CLIPDROP->value);

        $driver = Entity::driver(EntityEnum::tryFrom($defaultPsEngine))->inputImageCount(1)->calculateCredit();

        $driver->redirectIfNoCreditBalance();

        if ($defaultPsEngine == EntityEnum::CLIPDROP->value) {
            $photo = $this
                ->service
                ->setAction($request->input('action'))
                ->setPhoto($request->file('photo'))
                ->generate();

            if (is_string($photo)) {
                PhotoStudio::query()->create([
                    'user_id' => auth()->id(),
                    'photo'   => $photo,
                    'payload' => $request->input('action'),
                    'status'  => 'completed',
                    'credits' => 1,
                ]);

                $driver->decreaseCredit();
            }

            if ($request->ajax()) {
                return response()->json([
                    'type'    => is_string($photo) ? 'success' : 'error',
                    'photo'   => $photo,
                    'message' => is_string($photo) ? 'Photo generated successfully' : data_get($photo, 'message', 'Failed to generate photo'),
                ]);
            }

            if (is_string($photo)) {
                return redirect()->route('dashboard.user.photo-studio.index')->with([
                    'type'    => 'success',
                    'message' => 'Photo generated successfully',
                    'photo'   => $photo,
                ]);
            }

            return redirect()->route('dashboard.user.photo-studio.index')->with([
                'type'    => 'error',
                'message' => data_get($photo, 'message', 'Failed to generate photo'),
            ]);
        }

        if ($defaultPsEngine == EntityEnum::NOVITA->value) {
            $data = [
                'photo'       => $request->file('photo'),
                'mask_file'   => $request->file('mask_file'),
                'description' => $request->get('description'),
            ];

            $response = $this->novitaService->generate($request->input('action'), $data);

            if ($response['status'] == 'error') {
                return response()->json([
                    'type'    => 'error',
                    'message' => $response['message'],
                ]);
            }

            if (isset($response['task_id'])) {
                PhotoStudio::query()->create([
                    'user_id'    => auth()->id(),
                    'photo'      => $response['photo'],
                    'payload'    => $request->input('action'),
                    'status'     => $response['status'],
                    'request_id' => $response['task_id'],
                    'credits'    => 1,
                ]);

                $driver->decreaseCredit();
            }

            if (isset($response['task_id'])) {
                return response()->json([
                    'type'    => 'success',
                    'photo'   => $response['photo'],
                    'message' => 'Photo generated successfully',
                ]);
            }

            return redirect()->route('dashboard.user.photo-studio.index')->with([
                'type'    => 'error',
                'message' => 'Failed to generate photo',
            ]);
        }

        return redirect()->route('dashboard.user.photo-studio.index')->with([
            'type'    => 'error',
            'message' => 'Failed to generate photo',
        ]);
    }

    public function delete(PhotoStudio $photoStudio): RedirectResponse
    {
        if (Auth::id() === $photoStudio->getAttribute('user_id')) {
            $photoStudio->delete();
        }

        return redirect()->route('dashboard.user.photo-studio.index')->with([
            'type'    => 'success',
            'message' => 'Photo deleted successfully',
        ]);
    }
}
