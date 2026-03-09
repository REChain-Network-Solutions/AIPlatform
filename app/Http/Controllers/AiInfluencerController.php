<?php

namespace App\Http\Controllers;

use App\Concerns\HasErrorResponse;
use App\Enums\AiInfluencer\VideoStatusEnum;
use App\Helpers\Classes\Helper;
use App\Models\ExportedVideo;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use RuntimeException;
use Throwable;

class AiInfluencerController extends Controller
{
    use HasErrorResponse;

    // index
    public function index()
    {
        $exportedVideos = auth()->user()->exportedVideos()->whereNot('status', VideoStatusEnum::FAILED->value)->orderBy('created_at', 'desc')->get();

        return view('panel.user.ai_influencer.index', compact('exportedVideos'));
    }

    // delete exported video
    public function deleteExportedVideo(Request $request): JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $request->validate([
            'video_id' => 'required',
        ]);

        try {
            ExportedVideo::find($request->input('video_id'))->delete();

            return response()->json([
                'status' => 'success',
            ]);
        } catch (Throwable $th) {
            return $this->exceptionRes($th, 'Error happend while delete the exported video');
        }
    }

    // upload files
    public function uploadFiles(Request $request): JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $request->validate([
            'video_files'   => 'sometimes|array',
            'image_files'   => 'sometimes|array',
        ]);

        try {
            $fileDisk = 'uploads';
            $rootDir = 'ai-influencer-assets';
            $uploadedFiles = [
                'video_urls' => [],
                'image_urls' => [],
            ];

            $folderPath = public_path('uploads/ai-influencer-assets');
            if (! file_exists($folderPath)) {
                if (! mkdir($folderPath, 755, true) && ! is_dir($folderPath)) {
                    throw new RuntimeException(sprintf('Directory "%s" was not created', $folderPath));
                }
            }

            foreach ($request->file('video_files') ?? [] as $file) {
                $path = $file->store($rootDir, ['disk' => $fileDisk]);
                $url = Storage::disk($fileDisk)->url($path);
                $uploadedFiles['video_urls'][] = $url;
            }

            foreach ($request->file('image_files') ?? [] as $file) {
                $path = $file->store($rootDir, ['disk' => $fileDisk]);
                $url = Storage::disk($fileDisk)->url($path);
                $uploadedFiles['image_urls'][] = $url;
            }

            return response()->json([
                'status'  => 'success',
                'resData' => [
                    'uploadedFiles' => $uploadedFiles,
                ],
            ]);
        } catch (Throwable $th) {
            return $this->exceptionRes($th, 'file upload failed for Ai influencer');
        }
    }
}
