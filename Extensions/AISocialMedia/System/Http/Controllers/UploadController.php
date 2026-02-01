<?php

namespace App\Extensions\AISocialMedia\System\Http\Controllers;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;

class UploadController extends Controller
{
    public function __invoke(Request $request)
    {
        $request->validate([
            'image' => 'required|image|mimes:jpeg,png,jpg,gif,svg|max:6144',
        ]);

        $path = $request->file('image')->store(options: ['disk' => 'public']);

        return response()->json([
            'path' => $path,
            'url'  => asset('uploads/' . $path),
        ]);
    }
}
