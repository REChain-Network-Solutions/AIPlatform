<?php

namespace App\Http\Controllers;

use App\Models\UserUpload;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Symfony\Component\HttpFoundation\Response;

class UserUploadController extends Controller
{
    public function show(Request $request, UserUpload $userUpload): Response
    {
        if ($request->user()?->id !== $userUpload->user_id
            || ! Storage::disk($userUpload->disk)->exists($userUpload->path)) {
            return response('', 404);
        }

        return Storage::disk($userUpload->disk)->download(
            $userUpload->path,
            $userUpload->original_name,
            [
                'Content-Type'              => $userUpload->mime_type,
                'X-Content-Type-Options'    => 'nosniff',
                'Content-Security-Policy'   => "default-src 'none'; sandbox",
            ],
        );
    }
}
